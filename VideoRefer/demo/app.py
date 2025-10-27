import spaces
import gradio as gr
import numpy as np
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import os
import cv2
import argparse
import sys
# This is for making model initialization faster and has no effect since we are loading the weights
sys.path.append('./')
from videollama3 import disable_torch_init, model_init, mm_infer, get_model_output
from videollama3.mm_utils import load_images
from videollama3.mm_utils import load_video


color_rgb = (1.0, 1.0, 1.0)
color_rgbs = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    ]

def extract_first_frame_from_video(video):
    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    cap.release()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def extract_points_from_mask(mask_pil):
    mask = np.asarray(mask_pil)[..., 0]
    coords = np.nonzero(mask)
    coords = np.stack((coords[1], coords[0]), axis=1)

    return coords

def add_contour(img, mask, color=(1., 1., 1.)):
    img = img.copy()

    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=8)

    return img

@spaces.GPU(duration=120)
def generate_masks(image, mask_list, mask_raw_list):
    image['image'] = image['background'].convert('RGB')
    # del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    mask = Image.fromarray((np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8) * 255).convert('RGB')
    points = extract_points_from_mask(mask)
    np.random.seed(0)
    if points.shape[0] == 0:
        raise gr.Error("No points selected")

    points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
    points = points[points_selected_indices]
    coords = [points.tolist()]
    mask_np = apply_sam(image['image'], coords)

    mask_raw_list.append(mask_np)
    mask_image = Image.fromarray((mask_np[:,:,np.newaxis] * np.array(image['image'])).astype(np.uint8))
    
    mask_list.append((mask_image, f"<region{len(mask_list)}>"))
    # Return a list containing the mask image.
    image['layers'] = []
    image['composite'] = image['background']
    return mask_list, image, mask_list, mask_raw_list

@spaces.GPU(duration=120)
def generate_masks_video(image, mask_list_video, mask_raw_list_video):
    image['image'] = image['background'].convert('RGB')
    # del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    mask = Image.fromarray((np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8) * 255).convert('RGB')
    points = extract_points_from_mask(mask)
    np.random.seed(0)
    if points.shape[0] == 0:
        raise gr.Error("No points selected")

    points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
    points = points[points_selected_indices]
    coords = [points.tolist()]
    mask_np = apply_sam(image['image'], coords)

    mask_raw_list_video.append(mask_np)
    mask_image = Image.fromarray((mask_np[:,:,np.newaxis] * np.array(image['image'])).astype(np.uint8))
    
    mask_list_video.append((mask_image, f"<object{len(mask_list_video)}>"))
    # Return a list containing the mask image.
    image['layers'] = []
    image['composite'] = image['background']
    return mask_list_video, image, mask_list_video, mask_raw_list_video


@spaces.GPU(duration=120)
def describe(image, mode, query, masks):
    # Create an image object from the uploaded image
    # print(image.keys())

    image['image'] = image['background'].convert('RGB')
    # del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    # Handle both hex and rgba color formats
    
    img_np = np.asarray(image['image']).astype(float) / 255.
    if mode=='Caption':
        mask = Image.fromarray((np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8) * 255).convert('RGB')
        
        points = extract_points_from_mask(mask)

        np.random.seed(0)

        if points.shape[0] == 0:
            if len(masks)>1:
                raise gr.Error("No points selected")

        else:
            # Randomly sample 8 points from the mask
            # Follow DAM https://github.com/NVlabs/describe-anything
            points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
            points = points[points_selected_indices]

            coords = [points.tolist()]

            mask_np = apply_sam(image['image'], coords)
            
            masks = []
            masks.append(mask_np)
        mask_ids = [0]
        
        img_with_contour_np = add_contour(img_np, mask_np, color=color_rgb)
        img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.).astype(np.uint8))
    else:
        img_with_contour_np = img_np.copy()

        mask_ids = []
        for i, mask_np in enumerate(masks):
            # img_with_contour_np = add_contour(img_with_contour_np, mask_np, color=color_rgbs[i])
            # img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.).astype(np.uint8))
            img_with_contour_pil = Image.fromarray((img_with_contour_np* 255.).astype(np.uint8))
            mask_ids.append(0)
    
    masks = np.stack(masks, axis=0)
    masks = torch.from_numpy(masks).to(torch.uint8)


    
    img = np.asarray(image['image'])
    

    if mode == "Caption":
        query = '<image>\nPlease describe the <region> in the image in detail.'
    else:
        if len(masks)==1:
            prefix = "<image>\nThere is 1 region in the image: <region0> <region>. "
        else:
            prefix = f"<image>\nThere is {len(masks)} region in the image: "
            for i in range(len(masks)):
                prefix += f"<region{i}><region>, "
            prefix = prefix[:-2]+'. '
        query = prefix + query
    # print(query)

    image['layers'] = []
    image['composite'] = image['background']

    text = ""
    yield img_with_contour_pil, text, image
    
    for token in get_model_output(
        [img],
        query,
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        mask_ids=mask_ids,
        modal='image',
        image_downsampling=1,
        streaming=True,
    ):
        text += token
        yield gr.update(), text, gr.update()

  
def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Could not read the video file.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)  
    return image

@spaces.GPU(duration=120)
def describe_video(video_path, mode, query, annotated_frame, masks, mask_list_video):
    # Create a temporary directory to save extracted video frames
    cap = cv2.VideoCapture(video_path)

    video_tensor = load_video(video_path, fps=4, max_frames=768, frame_ids=[0])

    annotated_frame['image'] = annotated_frame['background'].convert('RGB')

    # Process the annotated frame from the image editor
    if isinstance(annotated_frame, dict):
        # Get the composite image with annotations
        frame_img = annotated_frame.get("image", annotated_frame.get("background"))
        if frame_img is None:
            raise gr.Error("No valid annotation found in the image editor.")
        frame_img = frame_img.convert("RGB")
        
        # Get the annotation layer
        if "layers" in annotated_frame and len(annotated_frame["layers"]) > 0:
            mask = Image.fromarray((np.asarray(annotated_frame["layers"][0])[..., 3] > 0).astype(np.uint8) * 255).convert("RGB")
        else:
            mask = Image.new("RGB", frame_img.size, 0)
    else:
        frame_img = annotated_frame.convert("RGB")
        mask = Image.new("RGB", frame_img.size, 0)

    img_np = np.asarray(annotated_frame['image']).astype(float) / 255.
    # Extract points from the annotated mask (using the first channel)
    if mode == "Caption":
        points = extract_points_from_mask(mask)
        np.random.seed(0)
        if points.shape[0] == 0:
            raise gr.Error("No points were selected in the annotation.")
        # Randomly select up to 8 points
        # Follow DAM https://github.com/NVlabs/describe-anything
        points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
        points = points[points_selected_indices]

        # print(f"Selected points (to SAM): {points}")

        coords = [points.tolist()]

        mask_np = apply_sam(annotated_frame['image'], coords)
    
        masks = []
        masks.append(mask_np)
        mask_ids = [0]

        # img_with_contour_np = add_contour(img_np, mask_np, color=color_rgb)
        # img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.).astype(np.uint8))


    else:
        img_with_contour_np = img_np.copy()
        
        mask_ids = []
        for i, mask_np in enumerate(masks):
            # img_with_contour_np = add_contour(img_with_contour_np, mask_np, color=color_rgbs[i])
            # img_with_contour_pil = Image.fromarray((img_with_contour_np * 255.).astype(np.uint8))
            mask_ids.append(0)
    


    masks = np.stack(masks, axis=0)
    masks = torch.from_numpy(masks).to(torch.uint8)


    

    if mode == "Caption":
        query = '<video>\nPlease describe the <region> in the video in detail.'
    else:
        if len(masks)==1:
            prefix = "<video>\nThere is 1 object in the video: <object0> <region>. "
        else:
            prefix = f"<video>\nThere is {len(masks)} objects in the video: "
            for i in range(len(masks)):
                prefix += f"<object{i}><region>, "
            prefix = prefix[:-2]+'. '
        query = prefix + query
    
    # Initialize empty text
    # text = description_generator
    annotated_frame['layers'] = []
    annotated_frame['composite'] = annotated_frame['background']

    if mode=="Caption":
        mask_list_video = []
        mask_image = Image.fromarray((mask_np[:,:,np.newaxis] * np.array(annotated_frame['image'])).astype(np.uint8))
        mask_list_video.append((mask_image, f"<object{len(mask_list_video)}>"))
    text = ""
    yield frame_img, text, mask_list_video, mask_list_video

    for token in get_model_output(
        video_tensor,
        query,
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        mask_ids=mask_ids,
        modal='video',
        streaming=True,
    ):
        text += token
        yield gr.update(), text, gr.update(), gr.update()


@spaces.GPU(duration=120)
def apply_sam(image, input_points):
    inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np


def clear_masks():
    return [], [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoRefer gradio demo")
    parser.add_argument("--model-path", type=str, default="DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B", help="Path to the model checkpoint")
    parser.add_argument("--prompt-mode", type=str, default="focal_prompt", help="Prompt mode")
    parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p for sampling")

    args_cli = parser.parse_args()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber")) as demo:

        mask_list = gr.State([])  
        mask_raw_list = gr.State([])  
        mask_list_video = gr.State([])  
        mask_raw_list_video = gr.State([])  


        HEADER = ("""
            <div>
                <h1>VideoRefer X VideoLLaMA3 Demo</h1>
                <h5 style="margin: 0;">Feel free to click on anything that grabs your interest!</h5>
                <h5 style="margin: 0;">If this demo please you, please give us a star ⭐ on Github or 💖 on this space.</h5>
            </div>
            </div>
            <div style="display: flex; justify-content: left; margin-top: 10px;">
            <a href="https://arxiv.org/pdf/2501.00599"><img src="https://img.shields.io/badge/Arxiv-2501.00599-ECA8A7" style="margin-right: 5px;"></a>
            <a href="https://github.com/DAMO-NLP-SG/VideoRefer"><img src='https://img.shields.io/badge/Github-VideoRefer-F7C97E' style="margin-right: 5px;"></a>
            <a href="https://github.com/DAMO-NLP-SG/VideoLLaMA3"><img src='https://img.shields.io/badge/Github-VideoLLaMA3-9DC3E6' style="margin-right: 5px;"></a>
            </div>
            """)

        with gr.Row():
            with gr.Column():
                gr.HTML(HEADER)
    

        image_tips = """
                ### 💡 Tips:

                🧸 Upload an image, and you can use the drawing tool✍️ to highlight the areas you're interested in.
            
                🔖 For single-object caption mode, simply select the area and click the 'Generate Caption' button to receive a caption for the object.
                
                🔔 In QA mode, you can generate multiple masks by clicking the 'Generate Mask' button multiple times. Afterward, use the corresponding object id to ask questions.
                
                📌 Click the button 'Clear Masks' to clear the current generated masks.
                
                """
        
        video_tips = """
                ### 💡 Tips:
                ⚠️ For video mode, we only support masking on the first frame in this demo.

                🧸 Upload an video, and you can use the drawing tool✍️ to highlight the areas you're interested in the first frame.
            
                🔖 For single-object caption mode, simply select the area and click the 'Generate Caption' button to receive a caption for the object.
                
                🔔 In QA mode, you can generate multiple masks by clicking the 'Generate Mask' button multiple times. Afterward, use the corresponding object id to ask questions.
                
                📌 Click the button 'Clear Masks' to clear the current generated masks.
                
                """
  

        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.ImageEditor(
                        label="Image",
                        type="pil", 
                        sources=['upload'], 
                        brush=gr.Brush(colors=["#ED7D31"], color_mode="fixed", default_size=10),
                        eraser=True,
                        layers=False,
                        transforms=[],
                        height=300,
                    )
                    generate_mask_btn = gr.Button("1️⃣ Generate Mask", visible=False, variant="primary")
                    mode = gr.Radio(label="Mode", choices=["Caption", "QA"], value="Caption")
                    query = gr.Textbox(label="Question", value="What is the relationship between <region0> and <region1>?", interactive=True, visible=False)
                    
                    submit_btn = gr.Button("Generate Caption", variant="primary")
                    submit_btn1 = gr.Button("2️⃣ Generate Answer", variant="primary", visible=False)
                    gr.Examples([f"./demo/images/{i+1}.jpg" for i in range(8)], inputs=image_input, label="Examples")
    
                with gr.Column():
                    mask_output = gr.Gallery(label="Referred Masks", object_fit='scale-down', visible=False)
                    output_image = gr.Image(label="Image with Mask", visible=True, height=400)
                    description = gr.Textbox(label="Output", visible=True)
                    
                    clear_masks_btn = gr.Button("Clear Masks", variant="secondary", visible=False)
            gr.Markdown(image_tips)

        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video")
                    # load_btn = gr.Button("🖼️ Load First Frame", variant="secondary")
                    first_frame = gr.ImageEditor(
                        label="Annotate First Frame",
                        type="pil", 
                        sources=['upload'], 
                        brush=gr.Brush(colors=["#ED7D31"], color_mode="fixed", default_size=10),
                        eraser=True,
                        layers=False,
                        transforms=[],
                        height=300,
                    )
                    generate_mask_btn_video = gr.Button("1️⃣ Generate Mask", visible=False, variant="primary")
                    gr.Examples([f"./demo/videos/{i+1}.mp4" for i in range(4)], inputs=video_input, label="Examples")

                with gr.Column():
                    mode_video = gr.Radio(label="Mode", choices=["Caption", "QA"], value="Caption")
                    mask_output_video = gr.Gallery(label="Referred Masks", object_fit='scale-down')

                    query_video = gr.Textbox(label="Question", value="What is the relationship between <object0> and <object1>?", interactive=True, visible=False)

                    submit_btn_video = gr.Button("Generate Caption", variant="primary")
                    submit_btn_video1 = gr.Button("2️⃣ Generate Answer", variant="primary", visible=False)
                    description_video = gr.Textbox(label="Output", visible=True)
                    
                    clear_masks_btn_video = gr.Button("Clear Masks", variant="secondary")

            gr.Markdown(video_tips)

        
        def toggle_query_and_generate_button(mode):
            query_visible = mode == "QA"
            caption_visible = mode == "Caption"
            return gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=caption_visible), gr.update(visible=caption_visible), [], "", [], [],[],[]

        video_input.change(load_first_frame, inputs=video_input, outputs=first_frame)

        mode.change(toggle_query_and_generate_button, inputs=mode, outputs=[query, generate_mask_btn, clear_masks_btn, submit_btn1, mask_output, output_image, submit_btn, mask_output, description, mask_list, mask_raw_list, mask_list_video, mask_raw_list_video])
        
        def toggle_query_and_generate_button_video(mode):
            query_visible = mode == "QA"
            caption_visible = mode == "Caption"
            return gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=query_visible), gr.update(visible=caption_visible), [], [], [], [], []


        mode_video.change(toggle_query_and_generate_button_video, inputs=mode_video, outputs=[query_video, generate_mask_btn_video, submit_btn_video1, submit_btn_video, mask_output_video, mask_list, mask_raw_list, mask_list_video, mask_raw_list_video])

        submit_btn.click(
            fn=describe,
            inputs=[image_input, mode, query, mask_raw_list],
            outputs=[output_image, description, image_input],
            api_name="describe"
        )

        submit_btn1.click(
            fn=describe,
            inputs=[image_input, mode, query, mask_raw_list],
            outputs=[output_image, description, image_input],
            api_name="describe"
        )

        generate_mask_btn.click(
            fn=generate_masks,
            inputs=[image_input, mask_list, mask_raw_list],
            outputs=[mask_output, image_input, mask_list, mask_raw_list]
        )

        generate_mask_btn_video.click(
            fn=generate_masks_video,
            inputs=[first_frame, mask_list_video, mask_raw_list_video],
            outputs=[mask_output_video, first_frame, mask_list_video, mask_raw_list_video]
        )

        clear_masks_btn.click(
            fn=clear_masks,
            outputs=[mask_output, mask_list, mask_raw_list]
        )

        clear_masks_btn_video.click(
            fn=clear_masks,
            outputs=[mask_output_video, mask_list_video, mask_raw_list_video]
        )

        submit_btn_video.click(
            fn=describe_video,
            inputs=[video_input, mode_video, query_video, first_frame, mask_raw_list_video, mask_list_video],
            outputs=[first_frame, description_video, mask_output_video, mask_list_video],
            api_name="describe_video"
        )

        submit_btn_video1.click(
            fn=describe_video,
            inputs=[video_input, mode_video, query_video, first_frame, mask_raw_list_video, mask_list_video],
            outputs=[first_frame, description_video, mask_output_video, mask_list_video],
            api_name="describe_video"
        )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    disable_torch_init()


    model, processor, tokenizer = model_init(args_cli.model_path)
    

    demo.launch(share=False)
