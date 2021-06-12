import detect_text, text_recog, shape_classification

dirname = './test_img/'
crop_files = detect_text.detect_text_img(dirname)
print(text_recog.img_text_recog(crop_files))
print(shape_classification.detect_pill_shape(dirname))