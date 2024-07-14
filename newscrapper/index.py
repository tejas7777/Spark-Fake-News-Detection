from prefect import flow, task
from scrapper import NewsScraper
from kafka_helper.producer import KafkaJSONProducer

@task
def scrape_news_task(scraper):
    return scraper.run()

@task
def send_to_kafka_task(data, kafka_producer):
    if data is None or len(data) == 0:
        return

    kafka_producer.send_json('news_topic', data)

@task
def close_kafka_connection(kafka_producer):
    kafka_producer.close()

@flow
def news_scraping_flow():
    scraper = NewsScraper('./config/news.json')
    kafka_producer = KafkaJSONProducer(['ec2-51-20-18-89.eu-north-1.compute.amazonaws.com:9092'])
    results = scrape_news_task(scraper)
    send_to_kafka_task(results, kafka_producer)
    close_kafka_connection(kafka_producer)

#TODO:
#Task 1
#--Add a Prefect task to send this data to Kafka.
#--This task should only run after News Scrapping task.
#--Run this tasks (scrape_news_task --> kafka tast) as a Prefect flow, which runs every hour.
#Task 2
#--Add more news sources to config and ensure nothing is breaking.


if __name__ == "__main__":
    news_scraping_flow.serve(
        name="newscrapper-deployment",
        interval=60
    )