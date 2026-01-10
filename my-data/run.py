from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

def crawl_job():
    print("크롤링 시작:", datetime.now())
    # 여기에 크롤링 로직
    print("크롤링 종료")

scheduler = BlockingScheduler()

scheduler.add_job(
    crawl_job,
    trigger="cron",
    hour=2,
    minute=0
)

print("스케줄러 시작")
scheduler.start()