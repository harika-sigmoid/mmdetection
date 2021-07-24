
def model_test(image):
    from mmdet.apis import init_detector, inference_detector, show_result_pyplot
    import mmcv
    # Load model
    config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'

    checkpoint_file = '/content/epoch_36.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Test a single image 
    # img = "/content/1.1 (1).jpg"
    img = image

    # Run Inference
    result = inference_detector(model, img)

    # Visualization results
    show_result_pyplot(img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.15)  
    return result

def coordinates(path,file,ind):
    from PIL import Image
    from PyPDF2 import PdfFileReader 
    from pdf2image import convert_from_path
    import decimal   
    from mmdet.Func import model_test
    result = model_test(path)
    im = Image.open(path,mode='r')
    x=im.size
    input1 = PdfFileReader(open(file, 'rb'))
    rect = input1.getPage(ind).mediaBox
    twod_list = []
    for i in range(len(result[0][2])):
        x1 = result[0][2][i][0]
        y1 = result[0][2][i][1]
        x2 = result[0][2][i][2]
        y2 = result[0][2][i][3]

        xa = decimal.Decimal(float(x1)) * decimal.Decimal(float((rect[2]/x[0])))
        ya = decimal.Decimal(float(rect[3])) - decimal.Decimal(float(y1)) * decimal.Decimal(float((rect[3]/x[1])))
        xb = decimal.Decimal(float(x2)) * decimal.Decimal(float((rect[2]/x[0])))
        yb = decimal.Decimal(float(rect[3])) - decimal.Decimal(float(y2)) * decimal.Decimal(float((rect[3]/x[1])))
        new=str(xa)+','+str(ya)+','+str(xb)+','+str(yb)
        twod_list.append(new)  
    return twod_list  
