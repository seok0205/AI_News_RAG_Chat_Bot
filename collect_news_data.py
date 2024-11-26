import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List, Dict
from datetime import datetime
import time
import os

'''AI times의 뉴스 목록 페이지의 HTML을 가져옴.'''
def get_news_list(page: int) -> str:
    url = "https://www.aitimes.com/news/articleList.html"
    params = {
        "page" : page,
        "view_type" : "sm"
    }
    headers = {
        'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"페이지 {page} 요청 중 에러 발생: {e}")
        return ""
    
'''기사 상세 페이지의 내용을 가져옴.'''
def get_article_content(url: str) -> str:
    headers = {
        'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 기사 본문 찾기
        article_content_div = soup.find("article", id="article-view-content-div")
        if article_content_div:
            # 모든 p 태그 찾기
            paragraphs = article_content_div.find_all("p")
            # 각 단락의 텍스트를 리스트로 모으기
            content_list = [p.text.strip() for p in paragraphs if p.text.strip()]
            # 단락들을 개행문자로 결합.
            return ''.join(content_list)
        return ""
    
    except requests.exceptions.RequestException as e:
        print(f"기사 내용 요청 중 에러 발생: {e}")
        return ""

'''HTML에서 뉴스 기사 정보를 파싱함.'''
def parse_news_info(html: str) -> List[Dict]:
    articles = []
    soup = BeautifulSoup(html, 'html.parser')
    content_list = soup.find("section", id="section-list")
    if not content_list:
        return articles
    news_items = content_list.find_all("li")
    print(f"찾은 {len(news_items)}개의 기사")

    for item in news_items:
        # 제목 추출
        title_elem = item.find("h4", class_="titles")
        title = title_elem.find("a").text.strip() if title_elem else ""

        # URL 추출
        url_elem = title_elem.find("a") if title_elem else None
        url = "https://www.aitimes.com" + url_elem["href"] if url_elem and "href" in url_elem.attrs else ""

        # 설명 추출
        desc_elem = item.find("p", class_="lead")
        description = desc_elem.find("a").text.strip() if desc_elem else ""

        # 날짜 추출
        date_elem = item.find("span", class_="byline")
        if date_elem:
            date = date_elem.find_all("em")[-1].text.strip()
        else:
            date = ""
        
        # 기사 본문 가져오기
        content = get_article_content(url) if url else ""
        article = {
            "title": title,
            "description": description,
            "url": url,
            "date": date,
            "content": content
            }
        
        articles.append(article)
        time.sleep(1)   # 서버 부하 감소 위한 시간 지연
    return articles

'''DF에서 가장 빠른 날짜, 늦은 날짜 추출'''
def get_date_range(df: pd.DataFrame) -> tuple:
    try:
        # 날짜 문자열을 datetime 객체로 변환.
        df['parsed_date'] = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M')
        
        # 가장 빠른 날짜, 늦은 날짜 추출.
        start_date = df['parsed_date'].min().strftime('%Y%m%d')
        end_date = df['parsed_date'].max().strftime('%Y%m%d')

        # 임시 컬럼 삭제
        df.drop('parsed_date', axis=1, inplace=True)
        return start_date, end_date
    except Exception as e:
        print(f"날짜 처리 중 에러 발생: {e}")
        return None, None
    
'''뉴스 기사 정보를 DF로 변환, JSON 파일로 저장'''
def save_to_files(articles: List[Dict], start_page: int, end_page: int, base_path: str = "./"):
    # DF 생성
    df = pd.DataFrame(articles)
    
    # 날짜 범위 추출
    start_date, end_date = get_date_range(df)
    
    if not (start_date and end_date):
        # 날짜 추출 실패시 현재 시간 사용
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"ai_times_news_p{start_page}-{end_page}_{current_time}.json"
    else:
        file_name = f"ai_times_news_{start_date}-{end_date}_p{start_page}-{end_page}.json"
    
    # DataFrame을 JSON 파일로 저장
    json_path = os.path.join(base_path, file_name)
    df.to_json(json_path, force_ascii=False, orient='records', indent=4)
    print(f"JSON 파일 저장 완료: {json_path}")
    
    # 데이터 수집 정보 출력
    print("\n[데이터 수집 정보]")
    print(f"수집 페이지: {start_page}-{end_page} 페이지")
    print(f"수집 기간: {start_date[:4]}.{start_date[4:6]}.{start_date[6:]} - {end_date[:4]}.{end_date[4:6]}.{end_date[6:]}")
    print(f"수집 기사: 총 {len(df)}건")
    
    # 데이터 미리보기
    print("\n[데이터 미리보기]")
    print(df.head())

def main():
    start_page = 1
    end_page = 5  # 수집할 마지막 페이지 번호
    page_unit = 10  # 한 번에 처리할 페이지 수
    download_folder = './ai_news' #저장 폴더 이름
    os.makedirs(download_folder, exist_ok=True) #폴더 생성
    print("뉴스 기사 정보 수집 중...")
    
    # 10페이지씩 처리
    for start_idx in range(start_page, end_page + 1, page_unit):
        all_articles = []
        end_idx = min(start_idx + page_unit - 1, end_page)
    
        print(f"\n=== {start_idx}~{end_idx} 페이지 수집 시작 ===")
    
        for page in range(start_idx, end_idx + 1):
            html = get_news_list(page)
            if html:
                page_articles = parse_news_info(html)
                all_articles.extend(page_articles)
                print(f"페이지 {page}: {len(page_articles)}개의 기사 정보 수집 완료")
    
        if not all_articles:
            print(f"{start_idx}~{end_idx} 페이지의 기사 정보를 가져오는데 실패했습니다.")
            continue
    
        print(f"\n{start_idx}~{end_idx} 페이지 : 총 {len(all_articles)}개의 기사 정보를 수집했습니다.")
    
        try:
            save_to_files(all_articles, start_idx, end_idx, download_folder)
        except Exception as e:
            print(f"데이터 저장 중 에러 발생: {e}")
    
        print(f"=== {start_idx}~{end_idx} 페이지 처리 완료 ===\n")
        
        # 다음 수집 전 잠시 대기
        time.sleep(0.1)

    print("\n모든 페이지 수집 완료!")
    
if __name__ == "__main__":
    main()