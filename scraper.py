import requests
import lxml
import csv
from bs4 import BeautifulSoup


def get_data(url):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
    }

    r = requests.get(url, headers)

    with open("data.html", "w", encoding="utf-8") as f:
        f.write(r.text)

    with open("data.html", encoding="utf-8") as f:
        src = f.read()

    soup = BeautifulSoup(src, "lxml")
    a = soup.find_all("a", class_="topictitle")

    topics_urls = []
    for link in a:
        topic_url = "https://khabmama.ru/forum" + link.get("href")
        topic_url = topic_url[:25] + topic_url[26:]
        topics_urls.append(topic_url)

    del topics_urls[0]

    messages = []
    for topic_url in topics_urls:
        r = requests.get(topic_url, headers)
        topic_name = topic_url.split("/")[-1]
        topic_name = topic_name[:9] + topic_name[10:]
        topic_name = topic_name.replace('?', '')
        topic_name = topic_name.replace('=', '')
        with open(f"Data/{topic_name}.html", "w", encoding="utf-8") as f:
            f.write(r.text)
        with open(f"Data/{topic_name}.html", encoding="utf-8") as f:
            src = f.read()
        soup = BeautifulSoup(src, "lxml")
        message = soup.find_all("div", class_="content")
        for data in message:
            messages.append(data.text.split('),'))
        with open("Data/topic.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(messages)


get_data('https://khabmama.ru/forum/viewforum.php?f=184')