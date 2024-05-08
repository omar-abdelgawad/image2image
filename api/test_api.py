import requests

image = open("test.png", "rb")
new_image = requests.post("http://127.0.0.1:5000/process_image", files={"image": image})
print(new_image.headers)
image.close()
