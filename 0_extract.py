import googleapiclient.discovery
import pandas as pd
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import os   

api_key = os.environ.get("YOUTUBE_API_KEY")

# ---------------------------------
# PULL ALL COMMENTS
# ---------------------------------

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = api_key

if not DEVELOPER_KEY:
    raise ValueError("CRITICAL ERROR: YOUTUBE_API_KEY environment variable is not set. Run: export YOUTUBE_API_KEY='your_key'")

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY
)

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part="snippet",
    videoId="-GJgqIJsTME",
    maxResults=10000
)

comments = []

# Execute the request.
response = request.execute()

# Get the comments from the response.
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

while (1 == 1):
  try:
   nextPageToken = response['nextPageToken']
  except KeyError:
   break
  nextPageToken = response['nextPageToken']
  # Create a new request object with the next page token.
  nextRequest = youtube.commentThreads().list(part="snippet", videoId="-GJgqIJsTME", maxResults=100, pageToken=nextPageToken)
  # Execute the next request.
  response = nextRequest.execute()
  # Get the comments from the next response.
  for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','public'])
df.info()

def scrape_10k_yt_comments(api_key, target_count=10000):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    # Using specific viral IDs that definitely have open comments
    video_ids = [
        "kJQP7kiw5Fk", # Despacito
        "9bZkp7q19f0", # Gangnam Style
        "OPf0YbXqDm0", # Uptown Funk
        "hT_nvWreIhg", # One Dance
    ]
    
    all_comments = []
    
    for v_id in video_ids:
        if len(all_comments) >= target_count:
            break
            
        next_page_token = None
        print(f"Scraping video: {v_id}...")
        
        while True:
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=v_id,
                    maxResults=100, 
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'author': comment['authorDisplayName'],
                        'updated_at': comment['updatedAt'],
                        'like_count': comment['likeCount'],
                        'text': comment['textDisplay'],
                        'video_id': v_id
                    })
                    
                    if len(all_comments) >= target_count:
                        return pd.DataFrame(all_comments)

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            except HttpError as e:
                # This catches the "comments disabled" error specifically
                if e.resp.status == 403:
                    print(f"Skipping {v_id}: Comments are disabled.")
                    break
                else:
                    raise e
                
    return pd.DataFrame(all_comments)

# --- EXECUTION ---
df_comments = scrape_10k_yt_comments(API_KEY, target_count=10000)

print(f"\nDone! Total rows collected: {len(df_comments)}")

#make csv
#block of code to figure out where it scraped tilll and to stop there, so we can scrape onward from that point

# ---------------------------------
# SAVE SCRAPED COMMENTS TO CSV
# ---------------------------------

# 1. Setup API Configuration
API_KEY = api_key  # <-- Paste your Google API Key here
youtube = build('youtube', 'v3', developerKey=api_key)

# 2. Target target settings to reach 10,000 rows
# Provide a list of highly discussed video IDs or popular video links
VIDEO_IDS = ['cYwioeHu_OU', 'Lfzu74XDyco', 'TiS6vnju_mI', 'QOcP5OvSwlI', 'dQw4w9WgXcQ'] 
TARGET_ROWS = 10000

all_comments = []

print("Starting YouTube Comments Scraping Pipeline...")

for video_id in VIDEO_IDS:
    if len(all_comments) >= TARGET_ROWS:
        break
        
    print(f"Scraping video: {video_id} ...")
    next_page_token = None
    
    while True:
        try:
            # Fetch comment threads
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Max allowed per page by Google API
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            
            # Extract data points
            for item in response.get('items', []):
                comment_data = item['snippet']['topLevelComment']['snippet']
                
                all_comments.append({
                    'video_id': video_id,
                    'comment_id': item['id'],
                    'author': comment_data.get('authorDisplayName'),
                    'comment_text': comment_data.get('textDisplay'),
                    'likes': comment_data.get('likeCount'),
                    'published_at': comment_data.get('publishedAt')
                })
                
                # Stop immediately if target row count is fulfilled
                if len(all_comments) >= TARGET_ROWS:
                    break
            
            if len(all_comments) >= TARGET_ROWS:
                print(f"Target of {TARGET_ROWS} rows reached!")
                break
                
            # Check if another page of comments exists for this video
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        except Exception as e:
            print(f"Error processing video {video_id} or API quota limit reached: {e}")
            break

# 3. Process into DataFrame and Export to CSV
df_scraped = pd.DataFrame(all_comments)

# Save file locally in your project folder
csv_filename = "youtube_comments_10k_v2.csv"
df_scraped.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print("\n--- Scraping Summary ---")
print(f"Total rows collected: {len(df_scraped)}")
print(f"File successfully saved as: {os.path.abspath(csv_filename)}")

# View top preview of your clean dataset
df_scraped.head()

response['items'][0]
df.head(10)

# ---------------------------------
# SORT BY LIKES AND GET TOP 10
# ---------------------------------

df.sort_values(by='like_count', ascending=False)[0:10]

# ---------------------------------
# PULL ALL COMMENTS FOR MULTIPLE VIDEOS
# ---------------------------------
# GET COMMENTS FUNCTION

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = api_key

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)


def getcomments(video):
  request = youtube.commentThreads().list(
      part="snippet",
      videoId=video,
      maxResults=100
  )

  comments = []

  # Execute the request.
  response = request.execute()

  # Get the comments from the response.
  for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
          comment['authorDisplayName'],
          comment['publishedAt'],
          comment['likeCount'],
          comment['textOriginal'],
          comment['videoId'],
          public
      ])

  while (1 == 1):
    try:
     nextPageToken = response['nextPageToken']
    except KeyError:
     break
    nextPageToken = response['nextPageToken']
    # Create a new request object with the next page token.
    nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100, pageToken=nextPageToken)
    # Execute the next request.
    response = nextRequest.execute()
    # Get the comments from the next response.
    for item in response['items']:
      comment = item['snippet']['topLevelComment']['snippet']
      public = item['snippet']['isPublic']
      comments.append([
          comment['authorDisplayName'],
          comment['publishedAt'],
          comment['likeCount'],
          comment['textOriginal'],
          comment['videoId'],
          public
      ])

  df2 = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','video_id','public'])
  return df2

df = getcomments('TYxqBTdOq24')
df

# FOR LOOP FOR LIST OF IDs
df = pd.DataFrame()
for i in ['QOcP5OvSwlI','Lfzu74XDyco','TiS6vnju_mI','cYwioeHu_OU']:
  df2 = getcomments(i)
  df = pd.concat([df, df2])

df

df['video_id'].value_counts()

