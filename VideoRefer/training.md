
### Data Format

If you want to train a referring video-llm on your data, you need to follow the procedures below to prepare the video/image sft data:

```json
[
    // video referring (single-frame-mask)
    {
        "id": 0,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat is <region> doing in the video?"
            },
            {
                "from": "gpt",
                "value": "...."
            },
            ...
        ],
        "annotation":[
            //object1
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
            },
            //object2
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
            },
            ...
        ]

    },
    // video referring (multi-frame-mask)
    {
        "id": 1,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat is <region> doing in the video?"
            },
            {
                "from": "gpt",
                "value": "...."
            },
            ...
        ],
        "annotation":[
            //object1
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                },
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
                ...
            },
            //object2
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                },
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
                ...
            },
            ...
        ]

    },
    //image referring 
    {
        "id": 0,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat is <region> doing in the video?"
            },
            {
                "from": "gpt",
                "value": "...."
            },
            ...
        ],
        "annotation":[
            //object1
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
            },
            //object2
            {
                "frame_idx":{
                    "segmentation": {
                        //rle format or polygon
                    }
                }
            },
            ...
        ]

    },
    //image understanding
    {
        "id": 2,
        "image": "images/xxx.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ],
    }
    // video understanding
    {
        "id": 3,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ],
    },
    ...
]
```