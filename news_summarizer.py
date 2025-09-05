import logging
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline
from bs4 import BeautifulSoup
import nltk
import re

# Basic logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class NewsSummarizer:
    def __init__(self):
        # Configure NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Initialize models
        try:
            # Summarization model
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # CPU
            )

            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=-1
            )

        except Exception as e:
            logging.error(f"Model initialization error: {e}")
            raise

    def get_news_sources(self):
        # Fox-only sources
        return {
            "RSS Feed World": "https://moxie.foxnews.com/google-publisher/world.xml",
            "RSS Feed Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
            "RSS Feed Science": "https://moxie.foxnews.com/google-publisher/science.xml",
            "RSS Feed Tech": "https://moxie.foxnews.com/google-publisher/tech.xml"
        }

    def fetch_articles(self):
        """
        Fetch articles from all configured news sources
        Returns a dictionary of {source_name: [article_urls]}
        """
        sources = self.get_news_sources()
        articles = {}

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {
                executor.submit(self._fetch_source, source, feed_url): (source, feed_url)
                for source, feed_url in sources.items()
            }

            for future in as_completed(future_to_source):
                source, feed_url = future_to_source[future]
                try:
                    source_articles = future.result()
                    articles[source] = source_articles
                    logging.info(f"Found {len(source_articles)} articles from {source}")
                except Exception as e:
                    logging.error(f"Error fetching {source}: {e}")

        return articles

    def _fetch_source(self, source_name, feed_url):
        """
        Fetch and parse articles from a single RSS feed
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml,text/xml,*/*;q=0.9'
        }

        try:
            response = requests.get(feed_url, headers=headers, timeout=15)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            article_urls = []
            for item in root.findall('.//item'):
                link = item.find('link')
                if link is not None and link.text:
                    article_urls.append(link.text)

            return article_urls

        except Exception as e:
            logging.error(f"Error fetching {source_name}: {e}")
            return []

    def process_articles(self, max_articles=5):
        """
        Fetch, summarize, and analyze sentiment of articles.
        Returns a list of {summary, sentiment}.
        """
        articles = self.fetch_articles()
        results = []

        for urls in articles.values():
            for url in urls[:max_articles]:
                try:
                    text = self._extract_article_text(url)
                    if not text or len(text.split()) < 50:
                        continue

                    # Break article into chunks within model limits
                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

                    summaries = []
                    for chunk in chunks:
                        try:
                            input_len = len(chunk.split())
                            # Dynamic summary length
                            max_len = max(5, int(input_len * 0.5))   # half of input length
                            min_len = max(3, int(input_len * 0.2))   # one-fifth of input length

                            s = self.summarizer(
                                chunk,
                                max_length=max_len,
                                min_length=min_len,
                                do_sample=False
                            )[0]['summary_text']
                            summaries.append(s)
                        except Exception as e:
                            logging.error(f"Summarization error: {e}")
                            continue

                    if not summaries:
                        continue

                    final_summary = " ".join(summaries)

                    sentiment = self.sentiment_analyzer(final_summary[:500])[0]

                    results.append({
                        "summary": final_summary,
                        "sentiment": sentiment['label']
                    })

                except Exception as e:
                    logging.error(f"Error processing article {url}: {e}")
                    continue

        return results

    def _extract_article_text(self, url):
        """
        Extract raw text from an article webpage.
        """
        try:
            response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            paragraphs = [p.get_text() for p in soup.find_all("p") if p.get_text()]
            article_text = " ".join(paragraphs)

            article_text = re.sub(r'\s+', ' ', article_text).strip()
            return article_text

        except Exception as e:
            logging.error(f"Error extracting text from {url}: {e}")
            return ""


if __name__ == "__main__":
    summarizer = NewsSummarizer()
    results = summarizer.process_articles(max_articles=3)

    for r in results:
        print("\nSummary:", r["summary"])
        print("Sentiment:", r["sentiment"])
