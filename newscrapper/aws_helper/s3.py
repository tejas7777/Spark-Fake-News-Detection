import boto3
import json
import os

class S3ConfigClient:
    def __init__(self):
        access_key = os.environ['ACCESS_KEY']
        secret_key = os.environ['SECRET_KEY']
        region_name = 'eu-north-1'

        self.session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.__s3 = self.session.resource('s3')

        self.__bucket_name = 'data-dynamo-news-feed-scrapper-config'
        self.__object_key = 'news.json'

    def read_json_config(self):
        obj = self.__s3.Object(self.__bucket_name, self.__object_key)
        data = obj.get()['Body'].read().decode('utf-8')
        print("[S3ConfigClient][read_json_config][success]")
        return json.loads(data)