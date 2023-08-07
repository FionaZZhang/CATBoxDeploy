from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import openai
import os
from werkzeug.utils import secure_filename
import random
from flask import send_from_directory
from io import BytesIO
from google.cloud import storage


app = Flask(__name__, static_folder='dist')
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = './uploads'
openai.api_key = os.environ.get("OPENAI_API_KEY")
current_caption = ''
current_language = ''
current_platform = ''
current_tags = ''
current_requirements = ''
user_caption = ''
current_tags_top = ''
session_id = 0

# Define the template prompts
prompts_english = [
    "1. A quote from a book or movie or celebrity, cite where it comes from.",
    "2. Using only emojis.",
    "3. An interesting word or sentence in an European language except English and Chinese, include a translation.",
    "4. A caption that you think is appropriate"
]

prompts_chinese = [
    "1. 书或电影或名人语录中的一句话",
    "2. 只用表情（emojis）",
    "3. 欧洲小语种的一个有意思的词或者一句话，给出中文翻译",
    "4. 一个你觉得合适的文案"
]

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load("mobilenet_v2.pth"))
model.eval()

# Load the class labels used by the pre-trained model
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels = requests.get(LABELS_URL).json()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


def upload_blob(source_file_name, destination_blob_name, folder):
    """Uploads a file to the bucket."""
    bucket_name = os.environ.get('BUCKET_NAME')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder}/{destination_blob_name}")
    blob.upload_from_filename(source_file_name)

def upload_blob_img(image_content, destination_blob_name, folder):
    """Uploads a file to the bucket."""
    bucket_name = os.environ.get('BUCKET_NAME')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder}/{destination_blob_name}")
    blob.upload_from_file(image_content, content_type='image/png')

def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket_name = os.environ.get('BUCKET_NAME')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def clear_uploads(session_folder):
    """Deletes a blob from the bucket."""
    bucket_name = os.environ.get('BUCKET_NAME')
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=f"uploads/{session_folder}")
    for blob in blobs:
        blob.delete()


def analyze_image(image_path):
    global current_tags_top
    image = Image.open(image_path)

    # Check if image is not jpg
    if image_path.split('.')[-1].lower() != 'jpg':
        # Change the path to jpg
        image_path = os.path.splitext(image_path)[0] + '.jpg'
        # Convert image to jpg and save it
        image.convert('RGB').save(image_path)
        # Open the new jpg image
        image = Image.open(image_path)

    image = image.resize((224, 224))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        _, indices = torch.topk(preds, 3)
        top_classes = [labels[idx] for idx in indices[0]]

        _, indices = torch.topk(preds, 2)
        top_one_classes = [labels[idx] for idx in indices[0]]

    tags = top_classes
    current_tags_top = top_one_classes

    return tags

def analyze_images(image_paths):
    all_tags = []
    for image_path in image_paths:
        image_tags = analyze_image(image_path)
        all_tags += image_tags
    return all_tags

def generate_caption(image_tags, requirements, caption, language, platform):
    # Prepare the prompt based on the language
    if language == "English":
        final_prompt = (
            "You are a social media influencer. You need to come up with some captions for making a social media post (do not include hashtags #). "
            f"Caption is for the platform {platform}. "
            f"Other requirements from the user: {requirements if requirements else 'None'}. "
            f"Caption provided by the user: {caption if caption else 'None'}. "
            "Here are the 4 captions that you need to generate (response in format no. Your Response): "
            f"{'; '.join(prompts_english)}. "
            "Here are the words describing the pictures, get the vibe not the actual words: "
            f"{', '.join(image_tags) if image_tags else 'None'}."
        )
    else:
        final_prompt = (
            "你是一位网红。你要给一些即将发送的图片想文案。不要在文案中加话题标签。"
            f"你要发送帖子的平台是 {platform}。"
            f"其他的要求: {requirements if requirements else '无'}。"
            f"用户提供的文案: {caption if caption else '无'}。"
            "请给出四个文案（回答请用排版：数字. 文案）: "
            f"{'; '.join(prompts_chinese)}。 "
            f"这里是描述图片的一些词： {', '.join(image_tags) if image_tags else '无'}。用中文回答。"
        )

    print(final_prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        max_tokens=200,
        messages=[
            {"role": "system", "content": final_prompt}
        ]
    )

    return response.choices[0].message["content"]


def generate_new_caption(index):
    print(current_language)
    if current_language == 'Chinese':
        prompt = (
            "你是一位网红。你要给一些即将发送的图片想文案。不要在文案中加话题标签。"
            f"你要发送帖子的平台是 {current_platform}。"
            f"其他的要求: {current_requirements if current_requirements else '无'}。"
            f"用户提供的文案: {user_caption if user_caption else '无'}。"
            f"请根据第{index}条要求给出一条文案（只要一条，回答模版请用 ’{index}. 你的回答‘）: "
            f"{'; '.join(prompts_chinese)}。 "
            f"这里是描述图片的一些词： {', '.join(current_tags) if current_tags else '无'}。用中文回答。"
        )
    else:
        prompt = (
            "You are a social media influencer. You need to come up with some captions for making a social media post (do not include hashtags #). "
            f"Caption is for the platform {current_platform}. "
            f"Other requirements from the user: {current_requirements if current_requirements else 'None'}. "
            f"Caption provided by the user: {user_caption if user_caption else 'None'}. "
            f"Please generate one single caption according to requirement number {index}: "
            f"{'; '.join(prompts_english)}. "
            "Here are the words describing the pictures, get the vibe not the actual words: "
            f"{', '.join(current_tags) if current_tags else 'None'}."
        )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        max_tokens=200,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    print(prompt)
    print(response.choices[0].message["content"])
    return response.choices[0].message["content"]


@app.route('/generate', methods=['POST'])
def generate():
    global current_caption
    global current_language
    global current_platform
    global current_tags
    global current_requirements
    global user_caption
    global session_id
    try:
        files = request.files.getlist('files')
        session_folder = None
        if files:
            # Create a unique subdirectory
            session_id = str(random.randint(10000, 99999))
            session_folder = session_id
            filepaths = []
            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
                #file.save(filepath)
                upload_blob(filename, filename, 'uploads/' + session_folder)  # upload file to GCS
                filepaths.append(filepath)
            image_tags = analyze_images(filepaths)
        else:
            image_tags = []

        requirements = request.form.get('requirements', '')
        caption = request.form.get('caption', '')
        language = request.form.get('language', 'English')
        platform = request.form.get('platform', 'Instagram')
        generated_caption = generate_caption(image_tags, requirements, caption, language, platform)
        user_caption = caption
        current_caption = generated_caption
        current_language = language
        current_platform = platform
        current_tags = image_tags
        current_requirements = requirements

        # clear the session folder after use
        if session_folder is not None:
            clear_uploads(session_folder)

        return jsonify({"caption": generated_caption}), 200
    except Exception as e:
        # clear the session folder in case of an error
        if session_folder is not None:
            clear_uploads(session_folder)
        return jsonify({"error": str(e)}), 500


@app.route('/regenerate', methods=['POST'])
def regenerate():
    try:
        global current_caption
        index = int(request.form.get('caption-index'))
        new_caption = generate_new_caption(index)
        print(new_caption)
        current_caption = new_caption
        return jsonify({"newcaption": new_caption}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_nft():
    try:

        if user_caption:
            prompts = f"NFT anime style art: {user_caption}"
        else:
            prompts = f"NFT anime style art: {current_tags_top[0]}, {current_tags_top[1]}"

        print(prompts)

        # Generate an image using the selected prompt and the DALL-E API
        image_response = openai.Image.create(
            prompt=prompts,
            n=1,
            size="256x256",
        )

        # Get the URL of the generated image
        image_url = image_response["data"][0]["url"]

        # Request the image content
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Save the image to a BytesIO object
        byte_data = BytesIO()
        img.save(byte_data, format='PNG')
        byte_data.seek(0)  # rewind to the start of the file

        # Upload the image to GCS
        upload_blob_img(byte_data, f'{session_id}.png', 'generated/')

        return jsonify({"image_url": image_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)