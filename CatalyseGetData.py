import numpy as np
import pandas as pd
import requests as rq
import json
from datetime import datetime, timedelta
import os
import serpapi
import time
import random


serp_key = r"4a517c80bf13cf1dc6ad67e1906627cce969b685d6ea53a8abeddb86ba9fe2e9"
rtoken = "Bearer clt.2.8dSERwmt0PlZAztO6VCyqOXLTRJr0nky7z9T97IP2bAC7iW3de2IC9SSIQgABv4OIw9ZUyQVx3UMRgmr6ylFOA*0"

# Google Trends

def gtrends(sdate, edate, terms):   
    serp_key = r"4a517c80bf13cf1dc6ad67e1906627cce969b685d6ea53a8abeddb86ba9fe2e9"
    client = serpapi.Client(api_key=serp_key)

    values = []

    search0 = client.search(
        engine="google_trends",
        q=terms,
        api_key=serp_key,
        geo="GB",
        date=sdate+" "+edate
    )

    # months = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    for entry in search0["interest_over_time"]["timeline_data"]:
        for term in entry["values"]:
            value = 0
            try:
                value += int(term["value"])
            except:
                pass
        values.append(value)

    return values


# tiktok

def get_weekly_date_ranges(start_date: str, end_date: str):
    """
    Generate weekly date ranges between start_date and end_date.
    
    Parameters:
    - start_date: str, starting date in 'yyyymmdd' format
    - end_date: str, ending date in 'yyyymmdd' format
    
    Returns:
    - List of tuples, where each tuple has (start_of_week, end_of_week) in 'yyyymmdd' format.
    """
    # Convert start and end dates to datetime objects
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    # List to store weekly date ranges
    weekly_ranges = []
    
    # Iterate from start to end date in weekly increments
    current_start = start
    while current_start <= end:
        # Define the end of the week as 6 days after the current start date
        current_end = current_start + timedelta(days=7)
        
        # Ensure the end date does not go beyond the specified final date
        if current_end > end:
            current_end = end
        
        # Format dates to 'yyyymmdd'
        start_str = current_start.strftime('%Y%m%d')
        end_str = current_end.strftime('%Y%m%d')
        
        # Add the (start, end) tuple to the list
        weekly_ranges.append((start_str, end_str))
        
        # Move to the next week
        current_start = current_start + timedelta(days=7)
    
    return weekly_ranges



def safe_post(url, headers, json, retries=3):
    for attempt in range(retries):
        resp = rq.post(url, headers=headers, json=json)
        if resp.status_code == 200 and resp.text.strip():
            try:
                return resp.json()
            except json.JSONDecodeError:
                pass  # invalid JSON, will retry
        elif resp.status_code >= 500:
            print(f"Server error {resp.status_code}, attempt {attempt+1}/{retries}...")
            time.sleep(2 ** attempt + random.random())  # exponential backoff
            continue
        else:
            raise RuntimeError(f"Fatal error {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError("Max retries reached, server still failing.")



def tiksearch(sdate, edate, terms, searchid=None, cursor=0):

    rtoken = "Bearer clt.2.ONmb-B9t3JY0aXdEr4-R9iHYrBYMVsa4p-vru1MMBDz95tdvtwp5WUw7ko4vLbnQAIBuSaO-DyECXcYYojOLqA*0"
    # URL with query fields
    url = "https://open.tiktokapis.com/v2/research/video/query/?fields=id,video_description,create_time"

    # Headers including authorization token and content type
    headers = {
        'Authorization': rtoken,
        'Content-Type': 'application/json'
    }

    # Data payload for the POST request
    data = {
        "query": {
            "and": [
                {
                    "operation": "EQ",
                    "field_name": "region_code",
                    "field_values": ["GB"]
                },
                {
                    "operation": "IN",
                    "field_name": "hashtag_name",
                    "field_values": terms
                }
            ]
        },
        'search_id': searchid,
        "max_count": 100,
        "cursor": cursor,
        "start_date": sdate,
        "end_date": edate
    }
    response = safe_post(url, headers=headers, json=data)
    data = response
    ids = []
    try:
        vids = data["data"]["videos"]
        for vid in vids:
            ids.append(vid)
        a = data["data"]["has_more"]
        b = data["data"]["search_id"]
        c = data["data"]["cursor"]
        f = 0
    except:
        a = False
        b = None
        c = 50
        f = 1
    return [[a,b,c,f], ids]

def toksearch(sD, eD, terms, folder):

    weeks = get_weekly_date_ranges(sD,eD)
    weeks = weeks[:-1]
    f = 0
    tscount = []
    for week in weeks:
        #print(week)
        all_rows = []
        ids = []
        a = False
        try: 
            ((a,b,c,f),vids) = tiksearch(sdate=week[0], edate=week[1], searchid=None, cursor=0, terms=terms)
            ids.extend(vids)
        except Exception as e:
            tscount.append(0)
            print("fail")
            print(e)
            f = 1
            return tscount, f
        while(a == True):
            try:
                ((a,b,c,f),vids) = tiksearch(sdate=week[0], edate=week[1], searchid=b, cursor=c, terms=terms)
                ids.extend(vids)
            except Exception as e:
                print("fail" + str(c))
                print(e)
                f = 1
                return tscount, f
        for vid in ids:
            vid["week"] = week
            all_rows.append(vid) 
        tscount.append(len(ids))
    
    with open(os.path.join(folder, "tiktokData.json"), 'w') as g:
        json.dump(all_rows, g)
    return tscount, f

                
            
            
            
            