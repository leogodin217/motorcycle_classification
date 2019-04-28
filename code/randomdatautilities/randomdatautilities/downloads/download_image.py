''' Functions to download resources'''
import requests
import os
import re
from uuid import uuid4  # Random uuids
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials
from pathlib import Path


def get_target_image(args):
    '''
    Downloads an image with a target

    :params
        args: A dict including the url, a pathlib.path object with the full
              folder path to save the image to and a target
            url: URL of the image
            path: pathlib.Path object describing where to save the image. (Must
                  include the image name)
            target: Target of the image. This is useful in machine learning. It
                    does not change logic, but is included in the return dict.
    :returns
        A dict with the url, result, saved path and target
    '''
    url = args['url']
    path = args['path']
    target = args['target']
    result = 'success'

    # get image name

    # Check for valid image
    headers = requests.head(url, allow_redirects=True)
    if headers.headers['Content-Type'].startswith('image/'):
        # It's an image. Let's download it
        image = requests.get(url, allow_redirects=True)
        if image.status_code == 200:
            with open(path.as_posix(), 'wb') as f:
                f.write(image.content)
        else:
            result = f'Status code: {image.status_code}'
    else:
        result = 'Not an image'
    return {'url': url, 'result': result, 'saved_path': path.as_posix(),
            'target': target}


def get_image_metadata(target):
    azure_key = os.environ['temp_azure_key']

    # Get the client
    client = ImageSearchAPI(CognitiveServicesCredentials(azure_key))

    base_save_path = Path('.').absolute() / f'../data/motorcycles/'

    # Search for each motorcycle class and obtain a list of urls and ensure
    # directories are created
    image_metadata = []
    target_path = re.sub(r' ', r'_', target)  # Used for the directory name
    download_path = base_save_path / target_path
    download_path.mkdir(parents=True, exist_ok=True)

    search_results = client.images.search(query=target)

    for index, image in enumerate(search_results.value[0:10]):
        name = str(uuid4()) + '-' + Path(image.content_url).name
        path = download_path / name

        image_metadata.append({
            'name': name,
            'url': image.content_url,
            'path': path,
            'target': target
        })

    return image_metadata
