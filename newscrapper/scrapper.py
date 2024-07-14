import feedparser # type: ignore
import json
from bs4 import BeautifulSoup # type: ignore
import hashlib
import redis # type: ignore
import hashlib
from kafka_helper.producer import KafkaProducer
from typing import Tuple
from aws_helper.s3 import S3ConfigClient

class NewsScraper:
    def __init__(self, config_path):
        self.config_path = config_path
        #self.config = self.load_rss_config()
        self.config = self.load_rss_config_s3()
        self.db = redis.Redis(host='localhost', port=6379, db=0)


    def load_rss_config(self):
        with open(self.config_path, 'r') as file:
            return json.load(file)
        
    def load_rss_config_s3(self):
        s3_config_client = S3ConfigClient()
        return s3_config_client.read_json_config()

    def fetch_rss_data(self, feed_urls):
        feeds = []
        for url in feed_urls:
            feed = feedparser.parse(url)
            feeds.append(feed)
        return feeds
    
    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=' ')
        return text
    
    def store_news_item(keydb, news_item):

        news_hash = hashlib.sha256(news_item['content'].encode()).hexdigest()

        if not keydb.sismember("processed_hashes", news_hash):
            print("Processing new item:", news_item['title'])
            keydb.sadd("processed_hashes", news_hash)
        else:
            print("Duplicate item skipped:", news_item['title'])

    def db_check(self, content) -> Tuple[bool, str]:
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        hash_key = f"hash:{content_hash}"
        
        if self.db.exists(hash_key):
            return (True, None)
        #else:
            #Expires it after 86400 seconds (24 hours)
            #self.db.setex(hash_key, 86400, 1)
            # return (False, hash_key)
        
        return (False, hash_key)
    
    def process_feed(self, feeds, label) -> list:
        processed_feeds = []

        for feed in feeds:
            for entry in feed.entries:
                try:
                    title = entry.title if entry.title != "" else "TITLE UNAVAILABLE"
 
                    if 'content' in entry and len(entry.content) > 0 and entry.content[0].value != "":
                        content = entry.content[0].value
                    elif 'summary' in entry and entry.summary != "":
                        content = entry.summary
                    else:
                        continue
                    is_duplicate, key = self.db_check(content)

                    if is_duplicate:
                        print(f"Skipping duplicate: {title}")
                        continue
                
                    clean_content = self.clean_html(content)
                    
                    published = entry.published if 'published' in entry else 'null'
                    
                    processed_feeds.append({
                        'title': title,
                        'text': clean_content,
                        'label': label,
                        'id': key[-5:],
                        'date': published
                    })
                    
                    self.db.setex(key, 86400, 1)

                except Exception as e:
                    print(f"[error][process_feed] {e}")
                

        return processed_feeds



    def run(self):
        real_news_feeds = self.process_feed( self.fetch_rss_data(self.config['news']['real']), 1 )
        fake_news_feeds = self.process_feed( self.fetch_rss_data(self.config['news']['fake']), 0 )

        return real_news_feeds + fake_news_feeds #Combine the Lists
    
        
        
