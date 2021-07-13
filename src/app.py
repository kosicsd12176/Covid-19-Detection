import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import time
from Covid_model import get_model
import torch
from torchvision import transforms



fig = plt.figure()

with open("../custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Covid-19 Chest X-ray Classifier')

st.markdown(
    "Welcome to this simple web application that classifies Covid-19 X-ray images")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions[0][0])
                st.pyplot(fig)


def predict(image):
    model_state = torch.load("/Users/k.stavrianos/PycharmProjects/Covid/mymodel2.pth")
    class_labels = ["Covid-19", "Normal"]
    model, _, _ = get_model()
    model.load_state_dict(model_state)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    #img = Image.open(image).convert('RGB')
    batch_t = torch.unsqueeze(transform(image.convert("RGB")), 0)

    out = model(batch_t)
    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(class_labels[idx], prob[idx].item()) for idx in indices[0][:1]]



if __name__ == "__main__":
    main()