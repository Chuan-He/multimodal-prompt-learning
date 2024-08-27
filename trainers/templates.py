# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

clip_template = {   'Caltech101':[
                    'a photo of a {}.',
                    'a painting of a {}.',
                    #'a plastic {}.',
                    'a sculpture of a {}.',
                    'a sketch of a {}.',
                    'a tattoo of a {}.',
                    #'a toy {}.',
                    'a rendition of a {}.',
                    #'a embroidered {}.',
                    #'a cartoon {}.',
                    #'a {} in a video game.',
                    #'a plushie {}.',
                    #'a origami {}.',
                    #'art of a {}.',
                    #'graffiti of a {}.',
                    'a drawing of a {}.',
                    'a doodle of a {}.',
                    'a photo of the {}.',
                    'a painting of the {}.',
                    #'the plastic {}.',
                    'a sculpture of the {}.',
                    'a sketch of the {}.',
                    'a tattoo of the {}.',
                    #'the toy {}.',
                    'a rendition of the {}.',
                    #'the embroidered {}.',
                    #'the cartoon {}.',
                    #'the {} in a video game.',
                    #'the plushie {}.',
                    #'the origami {}.',
                    #'art of the {}.',
                    #'graffiti of the {}.',
                    'a drawing of the {}.',
                    'a doodle of the {}.',
                    ], 
                    'DescribableTextures':[
                        'a photo of a {} texture.',
                        'a photo of a {} pattern.',
                        'a photo of a {} thing.',
                        'a photo of a {} object.',
                        'a photo of the {} texture.',
                        'a photo of the {} pattern.',
                        'a photo of the {} thing.',
                        'a photo of the {} object.',
                    ],
                    'EuroSAT':[
                        #'a centered satellite photo of {}.',
                        #'a centered satellite photo of a {}.',
                        #'a centered satellite photo of the {}.',
                    ], 
                    'FGVCAircraft':[
                        #'a photo of a {}.',
                        'a photo of a {}, a type of aircraft.',
                        #'a photo of the {}.',
                        #'a photo of the {}, a type of aircraft.',
                        #'a photo of one {}.',
                        #'a photo of one {}, a type of aircraft.',
                        #'a picture of a {}.',
                        'a picture of a {}, a type of aircraft.',
                        #'a picture of the {}.',
                        #'a picture of the {}, a type of aircraft.',
                        #'a picture of one {}.',
                        #'a picture of one {}, a type of aircraft.',
                        #'a drawing of a {}.',
                        #'a drawing of a {}, a type of aircraft.',
                        #'a drawing of the {}.',
                        #'a drawing of the {}, a type of aircraft.',
                        #'a drawing of one {}.',
                        #'a drawing of one {}, a type of aircraft.',
                    ],
                    'Food101': [
                        'a photo of {}, a type of food.',
                    ],
                    'OxfordFlowers':[
                        #'a photo of a {}',
                        #'a photo of the {}',
                        #'a picture of a {}',
                        #'a photo of the {}',
                        #'a picture of the {}',
                        #'a photo of a {}, a type of flower.',
                    ],
                    'OxfordPets':[
                        'a photo of a {}, a type of pet.',
                    ],
                    'StanfordCars':[
                        'a photo of a {}.',
                        'a photo of the {}.',
                        'a photo of my {}.',
                        #'i love my {}!',
                        #'a photo of my dirty {}.',
                        #'a photo of my clean {}.',
                        #'a photo of my new {}.',
                        #'a photo of my old {}.',
                    ],
                    'SUN397':[
                        'a photo of a {}.',
                        'a photo of the {}.',
                    ],
                    'UCF101':[
                        #'a photo of a person {}.',
                        #'a picture of a person {}.',
                        #'a photo of a person doing {}.',
                        #'a video of a person {}.',
                        #'a example of a person {}.',
                        #'a demonstration of a person {}.',
                        #'a photo of the person {}.',
                        #'a video of the person {}.',
                        #'a example of the person {}.',
                        #'a demonstration of the person {}.',
                        #'a photo of a person using {}.',
                        #'a video of a person using {}.',
                        #'a example of a person using {}.',
                        #'a demonstration of a person using {}.',
                        #'a photo of the person using {}.',
                        #'a video of the person using {}.',
                        #'a example of the person using {}.',
                        #'a demonstration of the person using {}.',
                        #'a photo of a person doing {}.',
                        #'a video of a person doing {}.',
                        #'a example of a person doing {}.',
                        #'a demonstration of a person doing {}.',
                        #'a photo of the person doing {}.',
                        #'a video of the person doing {}.',
                        #'a example of the person doing {}.',
                        #'a demonstration of the person doing {}.',
                        #'a photo of a person during {}.',
                        #'a video of a person during {}.',
                        #'a example of a person during {}.',
                        #'a demonstration of a person during {}.',
                        #'a photo of the person during {}.',
                        #'a video of the person during {}.',
                        #'a example of the person during {}.',
                        #'a demonstration of the person during {}.',
                        #'a photo of a person performing {}.',
                        #'a video of a person performing {}.',
                        #'a example of a person performing {}.',
                        #'a demonstration of a person performing {}.',
                        #'a photo of the person performing {}.',
                        #'a video of the person performing {}.',
                        #'a example of the person performing {}.',
                        #'a demonstration of the person performing {}.',
                        #'a photo of a person practicing {}.',
                        #'a video of a person practicing {}.',
                        #'a example of a person practicing {}.',
                        #'a demonstration of a person practicing {}.',
                        #'a photo of the person practicing {}.',
                        #'a video of the person practicing {}.',
                        #'a example of the person practicing {}.',
                        #'a demonstration of the person practicing {}.',
                    ],
    }

LASP_PROMPTS = {
       "a photo of a {}, a type of flower.", # flower
       "a photo of a person doing {}.",  #ucf101
       "a centered satellite photo of {}.", #eurosat
       "a photo of a {}, a type of aircraft.", # fgvc_aircraft
       "{} texture.", #dtd
       "itap of a {}.", #imagenet
       "a bad photo of the {}.", #imagenet
       "a origami {}.", #imagenet
       "a photo of the large {}.", #imagenet
       "a {} in a video game.", #imagenet
       "art of the {}.", #imagenet
       "a photo of the small {}.", #imagenet
       "a photo of a {}.",
       "a photo of many {}.", #imagenet
       "a photo of the hard to see {}.", #imagenet
       "a low resolution photo of the {}.", #imagenet
       "a rendering of a {}.", #imagenet
       "a bad photo of the {}.", #imagenet
       "a cropped photo of the {}.", #imagenet
       "a pixelated photo of the {}.", #imagenet
       "a bright photo of the {}.", #imagenet
       "a cropped photo of a {}.", #imagenet
       "a photo of the {}.", 
       "a good photo of the {}.",
       "a rendering of the {}.",
       "a close-up photo of the {}.",
       "a low resolution photo of a {}.",
       "a rendition of the {}.",
       "a photo of the clean {}.",
       "a photo of a large {}.",
       "a blurry photo of a {}.",
       "a pixelated photo of a {}.",
       "itap of the {}.",
       "a jpeg corrupted photo of the {}.",
       "a good photo of a {}.",
     }
template_dict = {   #'Caltech101': ["a photo of a {}.","this is a photo {}.","this is picture of {}.","one picture of a {}.", "a picture of a {}.","a painting of a {}.","a photo of the {}.","a painting of the {}."],
                    'Caltech101': ['a photo of a {}.','a painting of a {}.','a plastic {}.','a sculpture of a {}.','a sketch of a {}.','a tattoo of a {}.','a toy {}.','a rendition of a {}.'],
                    #'DescribableTextures':['a photo of a texture {}.', "this is a photo texture {}.","this is a picture texture {}.","one picture of a texture {}.",
                    #                       'a photo of a texture', "a photo of a pattern","a photo of a thing","a photo of a object"],
                    'DescribableTextures':['a photo of a {} texture.','a photo of a {} pattern.','a photo of a {} thing.','a photo of a {} object.',
                                           'a photo of the {} texture.','a photo of the {} pattern.','a photo of the {} thing.','a photo of the {} object.',],
                    'EuroSAT':['a centered satellite photo of {}.', 'a centered satellite picture of {}.','one centered satellite photo of a {}.','a satellite photo of a {}.',
                               'one centered satellite picture of {}.','a centered satellite photo of a {}.','this is centered satellite photo of {}.','a centered satellite photo of the {}.'], 
                    'FGVCAircraft':['this is aircraft picture of {}.','one picture of the aircraft {}.','a photo of the aircraft {}.','a photo of a {}.','a photo of the {}.'
                                    'a photo of a {}, a type of aircraft.', 'a photo of the {}, a type of aircraft.', 'this is aircraft photo of {}.'],
                    'Food101':['a photo of a {}.', 'this is a photo of {}.', 'this is a picture of {}.','one picture of a {}.',
                               'a photo of {}, a type of food.','a photo of the food {}.', 'this is photo of {}, a type of food.', 'this is picture of {}, a type of food.'],
                    #'ImageNet':["a photo of a","this is a photo", "itap of a", "a origami"],
                    'ImageNet':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetSketch':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetV2':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetA':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    'ImageNetR':["itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", 
                                "a {} in a video game.", "art of the {}.", "a photo of the small {}.", "a photo of a {}."],
                    #'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                    'OxfordFlowers':['a photo of a {}.', 'one picture of flower {}.',' a photo of flower {}.','this is photo of {}, a type of flower.',
                                     'a photo of a {}, a type of flower.','a picture of a {}, a type of flower.','a photo of the {}, a type of flower.','this is picture of {}, a type of flower.'],
                    #'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                    'OxfordPets':['a photo of a {}.','a photo of the {}.','one picture of a {}.', 'this is photo of {}.',
                                  'a photo of a {}, a type of pet.','a photo of the {}, a type of pet.','one picture of a {}, a type of pet.','this is photo of {}, a type of pet.'],
                    #'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a"],
                    'StanfordCars':['a photo of a {}.', 'a photo of the {}.', 'a photo of my {}.', 'i love my {}!', 
                                    'a photo of my dirty {}.', 'a photo of my clean {}.', 'a photo of my new {}.', 'a photo of my old {}.',],
                    'SUN397':["a photo of a {}.","this is photo of {}.","this is picture of {}.","one picture of a {}.",
                              'a photo of the {}.',"this is a photo of {}.","this is a picture of {}.","one picture of the {}."],
                    #'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing', 'a picture of a person doing', 'this is a photo of people doing'],
                    'UCF101':['a photo of a person {}.', 'a video of a person {}.', 'a example of a person {}.', 'a demonstration of a person {}.',
                                'a photo of the person {}.', 'a video of the person {}.', 'a example of the person {}.', 'a demonstration of the person {}.'],}

template_dict_plot = {'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a"], 
                    'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                    'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a'], 
                    'FGVCAircraft':['a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft'],
                    'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food'], 
                    'ImageNet':["a photo of a","this is a photo ","this is a","one picture of a"],
                    'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                    'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                    'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a"],
                    'SUN397':["a photo of a","this is a photo","this is picture of","one picture of a"],
                    'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing'],}