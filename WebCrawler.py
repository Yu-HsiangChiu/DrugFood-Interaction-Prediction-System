from icrawler.builtin import BingImageCrawler
import os

keywords = [
    "Asparagus", 
    "Averrhoa carambola", 
    "Banana", 
    "Beer", 
    "Broccoli", 
    "Cheese", 
    "Cured Meat", 
    "Ginger", 
    "Grapefruit slices", 
    "Mango", 
    "Orange", 
    "Papaya", 
    "Pomelo", 
    "Rice", 
    "Sausage", 
    "Spinach vegetable", 
    "Tomato"
]
limit = 50

for k in keywords:
    print(f"\n {k}...")
    
    folder_path = os.path.join("dataset", k)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    bing_crawler = BingImageCrawler(
        downloader_threads=2,             
        storage={'root_dir': folder_path} 
    )
    
    try:
        bing_crawler.crawl(
            keyword=k, 
            max_num=limit
        )
    except Exception as e:
        print(f"{k} error")

print("\n finish")
