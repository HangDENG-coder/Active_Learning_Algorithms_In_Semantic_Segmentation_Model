"""
Lambda to check whether an image is properly diluted. Also runnable as a script, to simplify testing.
"""

#This is a github test

import base64
import logging
import sys
import random

import numpy
import cv2

# "context" is an unused parameter but is required by the lambda API, so we'll disable the warning here:
# pylint: disable=W0613

BACKGROUND_MINIMUM = 0.15

def lambda_handler(event, context):
    """
    When we are called in the cloud, we enter through here. event is a JSON object with one member we care about:
    image, which is a base64-encoded JPEG file.
    """
    try:
        image = cv2.imdecode(numpy.frombuffer(base64.b64decode(event['image']), dtype=numpy.uint8), cv2.IMREAD_GRAYSCALE)
        prior_interval = event["priorInterval"] if "priorInterval" in event else None
        dilution_ok, background_ratio, _ = is_dilution_ok(image, prior_interval=prior_interval)
        dilution_ok, background_ratio, _ = is_dilution_ok(image)
        error = None
    except Exception as e:
        error = str(e)
        logging.exception(e)
        dilution_ok = False
        background_ratio = 0
    return {
        'result': dilution_ok,
        'backgroundRatio': background_ratio,
        'backgroundMinimum': BACKGROUND_MINIMUM,
        'error': error
    }

def is_dilution_ok(image, kernel_radius=1, threshold=0.05, prior_interval=None, squares_to_sample=100000, dilution_threshold=BACKGROUND_MINIMUM):
    """
    image: is a one channel grey scale numpy array of shape W x H with pixel values in [0,255]
    kernel_radius: corresponds to the subimage size ds = 2 * kernel_radius + 1 (we work with odd kernel size only)
      goes trough every pixel. if the pixel intensity is in interval it draws a square of size ds around the pixel
    threshold: the max-min variation sensitivity in the neighbourhood in order to assign it to the background
    prior_interval: is a prior assumption about the intensity range of the background. Hence, if the image is too
      bright or too dimmed this method will return is_dilution_ok=False
    squares_to_sample: the maximum number of pixels to check. This algorithm is not very efficient but due to the
      self-similarity of the images it is enough to check only a limited number of pixels and then extrapolate.
    dilution_threshold: The minimal needed background ratio below which we set dilution_ok=False
    """
    prior_interval = [0.15, 0.70] if prior_interval is None else prior_interval
    image = image / 255.
    image_shape = image.shape
    image_width = image_shape[0]
    image_height = image_shape[1]
    background_intensity_list = []
    background_pixels = 0
    for _ in range(squares_to_sample):
        x = random.randint(kernel_radius, image_width - kernel_radius - 1)
        y = random.randint(kernel_radius, image_height - kernel_radius - 1)
        if prior_interval[0] <= image[x, y] <= prior_interval[1]:
            sub_image = image[x - kernel_radius:x + kernel_radius + 1, y - kernel_radius:y + kernel_radius + 1]
            if numpy.max(sub_image) - numpy.min(sub_image) < threshold:
                background_pixels += 1
                background_intensity_list.append(numpy.mean(sub_image))

    background_ratio = background_pixels / squares_to_sample
    dilution_ok = background_ratio > dilution_threshold
    if len(background_intensity_list) < 1:
        print('The image is too dense and has absolutely no background!')
        # we set the background color to zero if no background was found (should be a rare case)
        background_intensity = 0
        dilution_ok = False
    else:
        background_intensity = numpy.mean(numpy.asarray(background_intensity_list))
    print(f'Background ratio = {background_ratio * 100:.2f}%, dilution_threshold = {dilution_threshold * 100}%, '
          f'dilution_ok = {dilution_ok}')
    return dilution_ok, background_ratio, background_intensity

def main(arguments):
    """
    Run the algorithm on a test image.
    :param arguments: The command line arguments
    """
    if len(arguments) < 2 or len(arguments) > 3:
        sys.stderr.write(f'Usage: {sys.argv[0]} <file>.jpg [expected]\n')
        sys.exit(1)
    image_file = arguments[1]
    try:
        expected = None if len(arguments) == 2 else {'true': True, 'false': False}[arguments[2].lower()]
    except KeyError:
        sys.stderr.write(f'Parameter "{arguments[2]}" must be "true" or "false"\n')
        sys.exit(1)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.stderr.write(f'Unable to read image "{image_file}"\n')
        sys.exit(1)
    result = is_dilution_ok(image)
    if expected is None:
        print(f'Image {image_file}: Dilution acceptable: {result}')
    elif result == expected:
        print(f'Image {image_file}: expected={expected}, actual={result}: Correct outcome')
    else:
        sys.stderr.write(f'Image {image_file}: expected={expected}, actual={result}: Incorrect outcome\n')
        sys.exit(1)

# When run as a command line, we analyze an image and print a result. The expected result can be provided, in which
# case we exit with an error code if the result does not match.
if __name__ == '__main__':
    main(sys.argv)
