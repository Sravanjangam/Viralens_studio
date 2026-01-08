import urllib.parse

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"


def generate_images(prompt: str, image_url: str = None, n: int = 4):
    """
    Generates multiple images using Pollinations.ai
    Returns list of image URLs
    """

    encoded_prompt = urllib.parse.quote(prompt)
    results = []

    for i in range(n):
        if image_url:
            url = (
                f"{POLLINATIONS_BASE}/{encoded_prompt}"
                f"?image={urllib.parse.quote(image_url)}"
                f"&seed={i}"
            )
        else:
            url = f"{POLLINATIONS_BASE}/{encoded_prompt}?seed={i}"

        results.append(url)

    return results
