import torch
import torch.nn as nn
import torch.nn.functional as F




def convert(predictions, s=7, B=2):
    '''
    params:
    predictions: batch_size x s x s x (C + 5B) tensor
    s x s: total grid cells
    B: bbox per grid cell
    bbox ->  (center x wrt grid, center y wrt grid, height wrt image, width wrt image)
    '''

    out = predictions.clone()
    centre = out.clone()
   # print('prediction:\n', out*100)
    # (center x wrt grid, center y wrt grid, height wrt image, width wrt image) to
    # (center x wrt image, center y wrt image, height wrt image, width wrt image) conversion
    cell_size_wrt_image = 1 / s
    for row in range(s):
        for col in range(s):
            for b in range(0, 5*B, 5):
                # centres from grid cell coord. to image coord. conversion
                centre[:, row, col, b+0] = (out[:, row, col, b+0] + col)  * cell_size_wrt_image
                centre[:, row, col, b+1] = (out[:, row, col, b+1] + row)  * cell_size_wrt_image
    #print('centre:\n', centre*100)
    
    # (center x, center y, height, width) attributes of bboxes, to 
    # (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    for b in range(0, 5*B, 5):
        out[..., b+0] = (centre[..., b+0] - centre[..., b+2] / 2)
        out[..., b+1] = (centre[..., b+1] - centre[..., b+3] / 2)
        out[..., b+2] = (centre[..., b+0] + centre[..., b+2] / 2) 
        out[..., b+3] = (centre[..., b+1] + centre[..., b+3] / 2)

    return out

def iou(box1, box2):
    '''
    Params:
    Intersection Over Union = Area of intersection / Area of union
    box: bounding box
          box1 == ... X 4  or ...x 4+
          box2 == ... X 4 (4 points)
          

    Returns: ... x 1 (ious)

    '''

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1, min=0) * torch.clamp(inter_rect_y2-inter_rect_y1, min=0)
 
    #Union Area
    b1_area = torch.abs((b1_x2 - b1_x1)*(b1_y2 - b1_y1))
    b2_area = torch.abs((b2_x2 - b2_x1)*(b2_y2 - b2_y1))
    
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-8)
    
    return iou

def nms(preds , conf_ph=0.1,iou_ph=0.1,S=7,B=2,C=20):
    '''
    preds (batch_size,s,s,30)
    conf_ph  - confidence threshhold
    iou_ph   -iou threshhold
    '''

    preds=convert(preds.cpu(),S,B)
    S=preds.shape[1]
    # print("iou in",preds[...,0:4].shape)
    # print("iou  return ",iou(preds[...,0:4],preds[...,5:9]).shape)
    # print("alladin iou  return ",intersection_over_union(preds[...,0:4],preds[...,5:9]).shape)

    out=[]
    for pred in preds:
        out.append(nms_(pred,conf_ph,iou_ph,B,C))
    return out

def nms_(preds, conf_ph=0.5,iou_ph=0.5,B=2,C=20):
    '''
    preds (s,s,30)
    conf_ph  - confidence threshhold
    iou_ph   -iou threshhold

    '''
    
    S=preds.shape[1]
  

    classes=[[]for _ in range(C)]
    for row in range(S):
        for col in range(S):
            ind1=torch.argmax(preds[row,col,5*B:])

            #conf thresholding
            for b in range(1,B+1):
                if preds[row,col,5*b-1] > conf_ph:
                    classes[ind1].append(preds[row,col,5*(b-1):5*b])
    out=[[]for _ in range(C)]
    for ind2,clas in enumerate(classes):
        if  len(clas)>0:
            temp=sorted(clas,key=lambda x:x[-1] ,reverse=True)
            b1=temp[0]
            #IOU  thresholding
            bboxes=[b1]
            print(b1.shape)
            if len(temp)>1:
                for b2 in(temp[1:]):
                    if iou(b1.unsqueeze(dim=0),b2.unsqueeze(dim=0)) < iou_ph:
                        bboxes.append(b2)
            out[ind2].append(bboxes)
    
    return out



def MAP():
    pass

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class loss(nn.Module):
    def __init__(self,S=7,B=2,C=20,lamda_cord=5,lamd_noobj=0.5):
        #dont  use inplace operators(+=) ,+ in intializing , intermediate nodes a=b+10
        super().__init__( )
        self.S=S
        self.B=B
        self.C=C
        self.lamda_cord=lamda_cord
        self.lamda_noobj=lamd_noobj

        #sum squared error
        #self.sse=nn.MSELoss(reduction='sum')


        super().__init__()
        pass

    def forward(self, prediction, target):
        '''prediction - bacth,S,S,5*B+C
           target     - batch,S,S,5+C

        '''
        Loss=0
        #for all bounding boxes per grid cell

        center_loss =torch.tensor(0.0)
        dim_loss    =torch.tensor(0.0)
        conf_loss   =torch.tensor(0.0)
        no_conf_loss=torch.tensor(0.0)
        class_loss  =torch.tensor(0.0)
        obj=target[:,:,:,4:5]
        #from center to diagonal points
        with torch.no_grad():
            pred_conv=convert(prediction)
            gt_conv  =convert(target,B=1)
        for batch in range(prediction.shape[0]):
            #SXS cell traversal
            for row in range(self.S):
                for col in range(self.S):
                    b_idx=0
                    with torch.no_grad():

                        box1=pred_conv[batch,row,col,0:5*self.B].view(self.B,5)
                        box2=gt_conv[batch,row,col,:5].repeat(self.B,1)

                        
                        b_idx=torch.argmax(iou(box1,box2))

                    

                
                    obj_present=target[batch,row,col,4]

                    if obj_present:



                        p_midx  = prediction[batch,row,col,5*b_idx:5*b_idx+1]
                        gt_midx = target[batch,row,col,0:1]

                        p_midy  = prediction[batch,row,col,5*b_idx+1:5*b_idx+2]
                        gt_midy = target[batch,row,col,1:2]
                        #centres loss
                        center_loss = center_loss+((p_midx - gt_midx)**2 + (p_midy - gt_midy)**2)
                       

                        p_w    = prediction[batch,row,col,5*b_idx+2:5*b_idx+3]
                        gt_w   = target[batch,row,col,2:3]

                        p_h    = prediction[batch,row,col,5*b_idx+3:5*b_idx+4]
                        gt_h   = target[batch,row,col,3:4]
                        #width hieght loss
                        dim_loss =dim_loss+( (torch.sign(p_w)*(torch.sqrt(torch.abs(p_w)+1e-8)) - torch.sqrt(gt_w))**2 
                                                    +
                                                    (torch.sign(p_h)*(torch.sqrt(torch.abs(p_h)+1e-8))   - torch.sqrt(gt_h))**2
                                                    
                                                    )


                        
                        p_c = prediction[batch,row,col,5*b_idx+4:5*b_idx+5]
                        g_c = target[batch,row,col,4:5]
                        #conf loss where there is object
                        conf_loss = conf_loss +(p_c-g_c)**2 
                                                        

                    else:
                        for bb in range(self.B):
                            p_c = prediction[batch,row,col,5*bb+4:5*bb+4+1]
                            g_c = target[batch,row,col,4:5]
                            #no conf loss  where there is no object ie.predction conf=0
                            no_conf_loss=no_conf_loss+(p_c-g_c)**2 
            
        #class loss of one hoted 
        class_loss = class_loss+ torch.sum(
                                          obj.repeat(1,1,1,self.C)*((prediction[:,:,:,5*self.B:] -target[:,:,:,5:])**2)
                                          )
            
        
        Loss= (
              self.lamda_cord*center_loss +
              self.lamda_cord*dim_loss +
              conf_loss +
              self.lamda_noobj*no_conf_loss +
              class_loss
              )

        print(self.lamda_cord*center_loss,self.lamda_cord*dim_loss,conf_loss,self.lamda_noobj*no_conf_loss,class_loss)
        print()
        return Loss



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import cv2

    yolo_loss=loss()
    pred=torch.randn((1,7,7,30),requires_grad=True)
    gt  =torch.ones((1,7,7,25))    
    l=yolo_loss(pred+10,gt)
    print(l)
    l.backward()
    print(l.item())




    # img = np.zeros((1000, 1000, 3), np.uint8)
    # s = 2
    # for i in range(0, s*500, 500):
    #     img[:, i] = 255
    #     img[i, :] = 255

    # pred = [[[[0.7, 0.5, .3, .45, 1.0], [0.5, 0.5, .2, .6, 1.0]],[[0.1, 0.3, .4, .2, 1.0], [0.1, 0.1, .5, .35, 1.0]]],]

    # pred = [[
    #         [[0.5, 0.6, .2, .6, 1.0], [0.5, 0.7, .2, .6, 1.0]],
    #         [[0.2, 0.3, .2, .2, 1.0], [0.1, 0.1, .5, .35, 1.0]]
    #        ],]

    # outo = convert(torch.tensor(pred), s=s, B=1)
    # out = outo.numpy() * 1000
    # out = out.astype(np.int32)
    # # print(out)

    # iu1 = iou(outo[:, 0, 0], outo[:, 1, 0])
    # iu2 = iou(outo[:, 0, 0], outo[:, 1, 1])
    # iu3 = iou(outo[:, 0, 1], outo[:, 1, 1])
    # print(iu1, iu2, iu3)

    # # draw boxes
    # for row in range(s):
    #     for col in range(s):
    #         box = out[0, row, col]
    #         print(box)
    #         cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), color=(0, 255, 255))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




# def IOU(bbox1, bbox2, i1, j1, i2, j2, s=2):
#     '''
#     Intersection Over Union = Area of intersection / Area of union
#     bbox: bounding box
#           batch_size x 5 tensor
#           [x, y, w, h, confidence P]
#           (x, y): centre of bbox wrt grid cell, b/n 0 to 1
#           (w, h): width and height of bbox wrt whole image, b/n 0 to 1
#     '''
#     cell_size_wrt_image = 1 / s
#     x1 = (bbox1[:, 0] + i1) * cell_size_wrt_image 
#     y1 = (bbox1[:, 1] + j1)* cell_size_wrt_image
#     w1 = bbox1[:, 2]
#     h1 = bbox1[:, 3]

#     x2 = (bbox2[:, 0] + i2 )* cell_size_wrt_image
#     y2 = (bbox2[:, 1] + j2) * cell_size_wrt_image
#     w2 = bbox2[:, 2]
#     h2 = bbox2[:, 3]

#     #box 1
#     x11=x1-w1/2
#     y11=y1-h1/2
#     x12=x1+w1/2
#     y12=y1+h1/2

#     #box2
#     x21=x2-w2/2
#     y21=y2-h2/2
#     x22=x2+w2/2
#     y22=y2+h2/2

#     b1_area=torch.abs(x12-x11)*torch.abs(y12-y11)
#     b2_area=torch.abs(x22-x21)*torch.abs(y22-y21)

#     int_x1,int_y1=torch.max(x11,x21),torch.max(y11,y21)
#     int_x2,int_y2=torch.min(x12,x22),torch.min(y12,y22)

#     intersection=(int_x2-int_x1).clamp(0)*(int_y2-int_y1).clamp(0)

#     print(intersection)
    

#     return intersection/(b1_area+b2_area-intersection+1e-8)

    


# def NMS():
#     pass

# def mAP():
#     pass



# class loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,pred,target):
#         pass




# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import matplotlib.patches as patches
# if __name__ == "__main__":

#     img=np.ones((448,448,3),dtype=np.uint8)*255


#     for i in range(0,448,224):
#         img[:,i] =0
#         img[i,:]=0

 

#     b1=torch.tensor([[0.5,0.6,0.2,0.6,0.9]])
#     b2=torch.tensor([[0.1,0.1,0.5,0.35,0.9]])

#     i1,j1,i2,j2=0,0,1,1

#     print(IOU2(b1,b2,i1,j1,i2,j2,2))

#     fig,ax = plt.subplots(1)
#     ax.imshow(img,cmap='gray')

#     # Create a Rectangle patch

#     w,h=b1[0][2]*448,b1[0][3]*448
#     x,y=(b1[0][0]+i1)*0.5*448-w/2,(b1[0][1]+j1)*0.5*448-h/2

    
#     rect1 = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')

#     w,h=b2[0][2]*448,b2[0][3]*448
#     x,y=(b2[0][0]+i2)*0.5*448-w/2,(b2[0][1]+j2)*0.5*448-h/2
    
#     rect2 = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')

#     ax.add_patch(rect1)
#     ax.add_patch(rect2)

#     plt.show()


#     pass
# %%
