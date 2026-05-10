from icrawler.builtin import GoogleImageCrawler
import os

queries = ["celebrity portrait face", "actor headshot", "person face photo"]
output_dir = r'C:\Users\Tugba Gul\Desktop\hedef_fotolar'

for query in queries:
    save_dir = os.path.join(output_dir, query.replace(" ", "_"))
    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=query, max_num=40)
    print(f"{query} tamamlandı.")
