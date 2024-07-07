import os
import requests
import time
import cv2
from ffpyplayer.player import MediaPlayer


def upload_img():
    # Upload a picture to the D-ID API
    # https://docs.d-id.com/reference/upload-an-image
    url = "https://api.d-id.com/images"

    cur_dir = os.path.dirname(__file__)
    img_path = os.path.join(cur_dir, "noelle.jpeg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"No such file or directory: '{img_path}'")
    
    files = { "image": ("noelle.jpeg", open(img_path, "rb"), "image/jpeg") }
    headers = {
        "accept": "application/json",
        "authorization": "Basic Y1hOak1USXlPVUJuYldGcGJDNWpiMjA6WDVnUklOWmdpY2dYM0RTcGN6cFZy" # YURJME1EZzBNREl5UUdkekxtNWphM1V1WldSMUxuUjM6NkZ6RUs2RHVvZFY2a0Q2WkRHYThr
    }
    response_pic_ = requests.post(url, files=files, headers=headers)
    # print(response_pic.text)
    return response_pic_

def create_new_talk(INPUT):
    # Create a new talk
    url = "https://api.d-id.com/talks"
    payload = {
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "microsoft",
                "voice_id": "zh-CN-XiaoxiaoNeural"
            },
            "input": INPUT
        },
        "config": {
            "fluent": "false",
            "pad_audio": "0.0"
        },
        # "webhook": "https://host.domain.tld/to/webhook",# Optional(不確定要怎麼用)
        "source_url": "s3://d-id-images-prod/google-oauth2|111061223712506778500/img_KlovwQ_m9xOaYa3qCXHHf/noelle.jpeg"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic Y1hOak1USXlPVUJuYldGcGJDNWpiMjA6WDVnUklOWmdpY2dYM0RTcGN6cFZy" # YURJME1EZzBNREl5UUdkekxtNWphM1V1WldSMUxuUjM6NkZ6RUs2RHVvZFY2a0Q2WkRHYThr
    }
    response_ = requests.post(url, json=payload, headers=headers)
    # print(response.text)
    return response_

def get_video(response_):
    # Get the video
    id = response_.json()["id"]
    url = f"https://api.d-id.com/talks/{id}"
    headers = {
        "accept": "application/json",
        "authorization": "Basic Y1hOak1USXlPVUJuYldGcGJDNWpiMjA6WDVnUklOWmdpY2dYM0RTcGN6cFZy" # YURJME1EZzBNREl5UUdkekxtNWphM1V1WldSMUxuUjM6NkZ6RUs2RHVvZFY2a0Q2WkRHYThr
    }
    response_video_ = requests.get(url, headers=headers)
    print(response_video_.text)
    return response_video_

def wait_for_video_completion(response_):
    elapsed_seconds = 0
    while True:
        response_video_ = get_video(response_)
        status = response_video_.json().get("status")
        
        if status == 'done':
            print(f"status: 'done', have been waited for {elapsed_seconds} second")
            url = response_video_.json().get("result_url")
            if url:
                return url
            else:
                print("Error: 'result_url' not found in response.")
                break
        
        time.sleep(1)
        elapsed_seconds += 1
        print(f"have been waited for: {elapsed_seconds} second, status: {status}")
    
    return None

def download_video(url):
    cur_dir = os.path.dirname(__file__)
    video_path = os.path.join(cur_dir, "..", "..", "files", "video", f"{url[-5:]}.mp4")
    response_ = requests.get(url, stream=True)
    if response_.status_code == 200:
        with open(video_path, 'wb') as file:
            for chunk in response_.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"video saved: {url[-5:]}.mp4")
    else:
        print(f"failed")
    return video_path

# def bumpup_vedio(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow('Video', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to quit the video playback
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     return

def bumpup_vedio(video_path):
    def get_audio_frame(player):
        audio_frame, val = player.get_frame()
        if val != 'eof' and audio_frame is not None:
            img, t = audio_frame
        return audio_frame, val

    video = cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)

    time.sleep(0.5)
    while True:
        grabbed, frame = video.read()
        audio_frame, val = get_audio_frame(player)
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)

    time.sleep(0.5)
    video.release()
    cv2.destroyAllWindows()
    
def main(response):
    response_pic_ = upload_img()
    print(response_pic_.text)

    response_ = create_new_talk(str(response))
    print(response_.text)

    url = wait_for_video_completion(response_)
    if url:
        print(f'\n{url}\n')
        video_path = download_video(url)
        bumpup_vedio(video_path)
    else:
        print("Failed to get video URL.")
    return

if __name__ == "__main__":
    response = input("\nPlease enter your speech...\n")
    main(response)