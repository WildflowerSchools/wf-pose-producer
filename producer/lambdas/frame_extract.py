from datetime import datetime
import logging
import os

import boto3
import cv2
import dateparser
from video_io import fetch_videos

from producer.helpers import parse_duration, ISO_FORMAT
from producer.lambdas.request_models import FrameExtractRequest


LOG_FORMAT = '%(levelname)-8s %(asctime)s %(processName)-12s %(module)14s[%(lineno)04d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def video_preloader_lambda_handler(event, _):  # context is _ because we don't use it
    print(os.environ['HONEYCOMB_CLIENT_ID'])
    request = FrameExtractRequest.from_dict(event)
    print(request)
    print("request to extract frames for %s starting at %s for %s", request.environment_name, request.timestamp, request.duration)
    start_date = dateparser.parse(request.timestamp)
    delta = parse_duration(request.duration)
    end_date = start_date + delta
    videos = _fetch_videos(request.environment_name, start_date, end_date)
    for obj in videos:
        obj["video_timestamp"] = obj["video_timestamp"].strftime(ISO_FORMAT)
    return {
        "videos": videos,
    }


def _fetch_videos(environment_name, start_date, end_date):
    results = fetch_videos(
        environment_name=environment_name,
        start=start_date,
        end=end_date,
        chunk_size=1000,
        local_video_directory='/mnt/data/prepared',
        video_filename_extension='mp4',
        download_workers=1,
        client_id=os.environ['HONEYCOMB_CLIENT_ID'],
        client_secret=os.environ['HONEYCOMB_CLIENT_SECRET']
    )
    print('videos prepared, downloaded %s videos', len(results))
    return results


def frame_extraction_lambda_handler(event, _):
    path = event.get("video_local_path")
    base = path[0:-4]
    logging.info("stareted %s", path)
    s3 = boto3.resource('s3', region_name="us-east-1")
    client = boto3.client('s3', region_name="us-east-1")
    bucket = s3.Bucket('wf-classroom-imagery')
    video_ts = datetime.strptime(event.get("video_timestamp"), ISO_FORMAT)
    prefix = f"frames/{event.get('environment_id')}/{video_ts.year}/{video_ts.month:02}/{video_ts.day:02}/{video_ts.hour:02}/{video_ts.minute:02}/{video_ts.second:02}/{event.get('device_id')}/"
    try:
        print(f"opening {path}")
        stream = cv2.VideoCapture(path)
        if stream.isOpened():
            frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
            files_on_s3 = client.list_objects_v2(
                Bucket='wf-classroom-imagery',
                Prefix=prefix,
                MaxKeys=int(frames*2),
            )
            if files_on_s3["KeyCount"] >= frames:
                print("all frames exist on s3 already")
                return
            print(f"video has {frames} frames")
            frame_num = 0
            while True:
                (grabbed, frame) = stream.read()
                if not grabbed:
                    break
                output_file_name = f"{base}-{frame_num:03}.jpg"
                cv2.imwrite(output_file_name, frame)
                key = f"{prefix}{frame_num:03}.jpg"
                bucket.upload_file(output_file_name, key)
                frame_num += 1
    except Exception as e:
        logging.exception(e)
    del stream
