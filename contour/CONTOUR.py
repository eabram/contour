### Contours
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import scipy.optimize
from skimage import data, color
from skimage.feature import canny
import itertools
from lmfit import models

sys.path.append('D:\\Documents\\Code\\pulse\\')

import utils
import ELPS

def get_image(filename,i_vec,j_vec,operator=None,cut=False,scale=1.0,fourier_shift=False,contour_shift=False,select=False,ang_deg = 0,blur_func=lambda im: im,contr_bright_startfunc=lambda im: im):
        # Load image
    if i_vec==[] and j_vec==[]:
        im = np.asarray(Image.open(filename).rotate(-ang_deg,))
    else:
        im = np.asarray(Image.open(filename).rotate(-ang_deg,center=(np.nanmean(j_vec),np.nanmean(i_vec)))) # rotates it back to proper alignment
    if cut!=False:
        A_calc = np.copy(im[:,:,:])
        if type(cut)==str:
            if cut=='SEM_scale':
                s1,s2,s3 = np.shape(A_calc)
                A_calc[s1-int(s2/17):,:,:] = np.ones((int(s2/17),s2,len(A_calc[0,0])))*np.nan
        else:
            A_calc[cut[0][0]:cut[0][1],cut[1][0]:cut[1][1],:] = np.ones((cut[0][1]-cut[0][0],cut[1][1]-cut[1][0],len(A_calc[0,0])))*np.nan
        im = np.copy(A_calc)
    #plt.figure()
    #plt.imshow(im)
    #plt.show()

    if len(i_vec)*len(j_vec)>0:
        di=abs(int(np.nanmax(i_vec[1:]-i_vec[0:-1])))
        dj=abs(int(np.nanmax(j_vec[1:]-j_vec[0:-1])))
        
        if fourier_shift==True:
            i_vec, j_vec = shift_by_fourier(im,i_vec,j_vec,operator,plot_on=False)

        Q = []
        for i in i_vec:
            for j in j_vec:
                Q.append(im[int(i-0.5*di*scale):int(i+0.5*di*scale),int(j-0.5*dj*scale):int(j+0.5*dj*scale)])
    else:
        if type(cut)!=str:
            Q = [im[300:2000-300,300:-300]]
        else:
            if cut=='SEM_scale':
                Q = [im[:s1-int(s2/17)]]
            else:
                Q = [im]

    if contour_shift==True:
        contrast = 1.0
        th1 = 60
        th2 = 40
        operator = lambda A: (A[:,:,0]*0.299+A[:,:,1]*0.587+A[:,:,0]*0.114)
        step=2
        edge_val = 1.0
        origin_val = 0.0
        pixmax = False
        out_sort = get_all_contours(Q[-1],operator,th1=th1,th2=th2,center_off=100,step=step,fitrun=False,inside=True,inside_try=False,edge_val=edge_val,origin_val=origin_val,pixmax=pixmax,circularity_min=0.0,split_contours=False,point_min=10,contrast=contrast)[0]
        c_center = out_sort[0]
        x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res,res_list = ELPS.run_ellipse(c_center)
        di = abs(i_vec[1]-i_vec[0])
        dj = abs(j_vec[1]-j_vec[0])
        i_vec_new = i_vec+int(-di*0.5+y0)
        j_vec_new = j_vec+int(-dj*0.5+x0)
        i_vec = i_vec_new
        j_vec = j_vec_new
        
        print('Shifted by: '+str(int(-di*0.5+y0))+', '+str(int(-dj*0.5+x0)))
        #...adjust for big shifts!!!
        Q = []
        for i in i_vec:
            for j in j_vec:
                Q.append(im[int(i-0.5*di*scale):int(i+0.5*di*scale),int(j-0.5*dj*scale):int(j+0.5*dj*scale)])
    
    #Q_tot = im[int(i_vec[0]-0.5*di*scale):int(i_vec[-1]+0.5*di*scale),int(j_vec[-1]-0.5*dj*scale):int(j_vec[0]+0.5*dj*scale)]
    Q_new = []
    for im in Q:
        Q_new.append(contr_bright_startfunc(blur_func(im)))
    Q = Q_new
    return Q

def shift_by_fourier(im,i_vec,j_vec,operator,plot_on=False):
    i_min = np.min(i_vec)
    i_max = np.max(i_vec)
    j_min = np.min(j_vec)
    j_max = np.max(j_vec)
    
    di = abs(i_vec[1]-i_vec[0])
    dj = abs(j_vec[1]-j_vec[0])
    
    scale = 1.0

    im_cut = operator(im[int(i_min-0.5*di*scale):int(i_max+0.5*di*scale),int(j_min-0.5*dj*scale):int(j_max+0.5*dj*scale)])
    
    isize = len(i_vec)
    jsize = len(j_vec)
    
    i_step = (i_max - i_min)/isize
    j_step = (j_max - j_min)/jsize
    
    qj = np.mean(im_cut,axis=0)
    qi = np.mean(im_cut,axis=1)
    
    I = utils.FFT(qi,pulse_per_second=1)[2]
    J = utils.FFT(qj,pulse_per_second=1)[2]
    
    phi_J = np.angle(J[jsize])
    phi_I = np.angle(I[isize])

    shifti = (phi_I*(i_max-i_min))/(isize*2*np.pi)
    shiftj = (phi_J*(j_max-j_min))/(jsize*2*np.pi)
    
    print(shifti,shiftj)
    
    si = (0.5*di-np.abs(shifti))*np.sign(shifti)
    sj = (0.5*dj-np.abs(shiftj))*np.sign(shiftj)

    print(si,sj)

    shifti = si
    shiftj = sj

    #shifti = int((np.sign(shifti)*0.5*di - shifti)*0.9)
    #shiftj = int((np.sign(shifti)*0.5*dj - shiftj)*0.9)
    #if np.abs(shifti)>0.1*di:
    #    shifti=0
    #elif np.abs(shiftj)>0.1*dj:
    #    shiftj=0
    
    print('Fourier Shifted by: ['+ str(shifti)+', '+str(shiftj)+']')
    i_vec_new = i_vec+shiftj
    j_vec_new = j_vec+shifti

    i_min = np.min(i_vec_new)
    i_max = np.max(i_vec_new)
    j_min = np.min(j_vec_new)
    j_max = np.max(j_vec_new)

    im_cut_new = operator(im[int(i_min-0.5*di*scale):int(i_max+0.5*di*scale),int(j_min-0.5*dj*scale):int(j_max+0.5*dj*scale)])
    
    if plot_on==True:
        f,ax = plt.subplots(1,2)
        ax[0].imshow(im_cut)
        ax[1].imshow(im_cut_new)
        plt.show()

    return i_vec+shifti, j_vec+shifti

def get_edges(image,operator,thresh=60,thresh_output=False,a=0.5,b=1,sigma=4.0,pixmax=False,lt=0.0,ht=0.2,clip_percent=1):
    # Load picture, convert to grayscale and detect edges
    image_gray = color.rgb2gray(image)
    A = operator(image)
    pixmedian = np.nanmedian(A)
    pixvar = np.nanvar(A)
    if pixmax==False:
        pixmax = np.nanmax(A)
        print('pixmax: ',pixmax)
        print()
    #print(pixmedian)
    #print(pixvar)
    #print()
    if clip_percent==0:
        edges = canny(abs(operator(image)-np.nanmedian(operator(image))), sigma=sigma,low_threshold=lt**pixmax, high_threshold=ht*pixmax)
    else:
        edges = canny(operator(image), sigma=sigma,
                  low_threshold=lt**pixmax, high_threshold=ht*pixmax)
    
    #convert img to grey
    img_grey = edges
    
    img_grey1 = image[:,:,0]
    img_grey2 = image[:,:,1]
    img_grey3 = image[:,:,2]
    if clip_percent==1:
        img_grey4 = operator(image)
    elif clip_percent==0:
        img_grey4 = operator(image)
        img_grey4 = abs(img_grey4-np.nanmedian(img_grey4))
        img_grey4 = (img_grey4*255)/np.nanmax(img_grey4)
    
    #img_grey_sel = ((edges*255*a)+img_grey1)/(1.0+a)
    img_grey_sel = np.array(((edges*255*a)+img_grey4*b)/(b+a),dtype=np.uint8)
    #img_grey_sel = threshold_by_background(image,operator,a=2,steps=100)
    #plt.figure()
    #plt.imshow(img_grey_sel)
    #plt.show()

    #set a thresh
    #thresh = 60
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey_sel, thresh, 255, cv2.THRESH_BINARY)
    #plt.figure()
    #plt.imshow(thresh_img)
    #plt.show()

    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #create an empty image for contours
    img_contours = np.zeros(image.shape)
    #plt.imshow(image)
    # draw the contours on the empty image
    for i, c in enumerate(contours):
        areaContour=cv2.contourArea(c)
        #if 10<areaContour:
        #    plt.plot(c[:,0,0],c[:,0,1],'r')
        #    continue
        #    #cv2.drawContours(image,contours,i,(255,10,255),4)
    
    #save image
    #cv2.imwrite('D:/contours.png',img_contours)
    if thresh_output==True:
        return img_grey_sel, contours, hierarchy,edges, thresh_img, pixmax
    elif thresh_output==False:
        return img_grey_sel, contours, hierarchy,edges, pixmax

def split_double_rim(c,lim=2,fit=True,fitval=None,ransac_opt=None):
    if fit==True:
        x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res, res_list = ELPS.run_ellipse(c,scale=1.0)
    elif fit==False:
        [x0,y0,ap,bp,e,phi] = fitval
    
    try:
        c[:,:,0]
        typ = 'cont'
        try:
            x
        except:
            x = c[:,:,0]
            y = c[:,:,1]
    except TypeError:
        typ = 'arr'
        [x,y] = c

    insidex = []
    insidey = []
    outsidex = []
    outsidey = []

    ins = []
    outs = []
    insidey = []
    outsidex = []
    outsidey = []

    ins = []
    outs = []

    for i in range(0,len(x)):
        inout = (((x[i]-x0)/(ap))**2)+(((y[i]-y0)/(bp))**2)
        if inout <0.98 and inout>0.8:
            insidex.append(x[i])
            insidey.append(y[i])
            ins.append(i)

        elif inout> 1.02 and inout<1.2:
            outsidex.append(x[i])
            outsidey.append(y[i])
            outs.append(i)
    if typ=='arr':
        cout = np.array([[outsidex],[outsidey]],dtype=np.int32).transpose()
        cin = np.array([[insidex],[insidey]],dtype=np.int32).transpose()

        return [cout,cin], True

    elif typ=='cont':
        cin = []
        for i in ins:
            cin.append(c[i][:][0:2])
        cin = np.array(cin)

        cout = []
        for i in outs:
            cout.append(c[i][:][0:2])
        cout = np.array(cout)


        #for i in range(0,len(x)):
        #    inout = (((x[i]-x0)/(ap))**2)+(((y[i]-y0)/(bp))**2)
        #    if inout <0.98 and inout>0.8:
        #        insidex.append(x[i])
        #        insidey.append(y[i])
        #        ins.append(i)

        #    elif inout> 1.02 and inout<1.2:
        #        outsidex.append(x[i])
        #        outsidey.append(y[i])
        #        outs.append(i)

        #cin = []
        #for i in ins:
        #    cin.append(c[i][:][0:2])
        #cin = np.array(cin)

        #cout = []
        #for i in outs:
        #    cout.append(c[i][:][0:2])
        #cout = np.array(cout)

        x1,y1,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1 = ELPS.run_ellipse_rs(cout,scale=1.0,ransac_opt=ransac_opt)
        x2,y2,xfit2,yfit2,[x02,y02,ap2,bp2,e2,phi2],res2, res_list2 = ELPS.run_ellipse_rs(cin,scale=1.0,ransac_opt=ransac_opt)

        if (((ap1-ap2)**2+(bp1-bp2)**2)**0.5)>lim:
            return [cout,cin], True
        else:
            return [c], False

def get_c_params(conts):
    cent_all = []
    area_all = []
    M_all = []
    circ_all = []
    for c in conts:
        if 'float' not in str(type(c)):
            #print(type(c))
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if cv2.arcLength(c,False)==0:
                circ_all.append(np.nan)
            else:
                circ_all.append(cv2.contourArea(c)/cv2.arcLength(c,False))

            try:
                cent = np.array([M["m10"] / M["m00"],M["m01"] / M["m00"]])
            except:
                cent = (np.array([np.inf,np.inf]))

        else:
            area = np.nan
            M = np.array([np.nan,np.nan])
            cent = np.array([np.nan,np.nan])
            circ = np.nan

        cent_all.append(cent)
        area_all.append(area)
        M_all.append(M)

    return cent_all, area_all, M_all, circ_all


def get_middle(rim_coor,params):
    [a,b] = params
    calc = 0
    for r in rim_coor:
        calc = calc+(((r[0]-a)**2+(r[1]-b)**2))
    return calc/len(rim_coor)

def get_angle(x,y):
    phi = np.arctan(np.abs(y/x))
    if x<0 and y>=0:
        phi = np.pi-phi
    elif x>=0 and y<0:
        phi = -phi
    elif x<0 and y<0:
        phi = np.pi + phi

    return phi

def ellipse_func(phi,a,b,alpha,Mx,My):
    theta = phi - alpha
    u = np.array([np.cos(alpha),np.sin(alpha)])
    v = np.array([-np.sin(alpha),np.cos(alpha)])
    X  = np.array([Mx,My]) + a*np.cos(theta)*u +(b*np.sin(theta))*v
    return X

def ellipse_calc(x,y,a,b,alpha,Mx,My,steps=101):
    phi_vec = np.linspace(0,2*np.pi,steps)
    X = np.array([ellipse_func(phi,a,b,alpha,Mx,My) for phi in phi_vec])
    calc = 0
    for i in range(0,len(x)):
        P = np.array([x[i],y[i]])
        calc = calc+np.nanmin([(np.linalg.norm(X[j]-P)**2) for j in range(0,len(X))])
    if len(x)!=0:
        return calc/len(x)
    else:
        return calc

def ellipse(x,y,a,b,alpha,Mx,My):
    # Minimize this function to obtain ellipse paramteres
    # Remark: Adjust by implementig true distance of point to ellipse
    calc = 0
    for i in range(0,len(x)):
        phi = get_angle(x[i]-Mx,y[i]-My)
        theta = phi - alpha
        u = np.array([np.cos(alpha),np.sin(alpha)])
        v = np.array([-np.sin(alpha),np.cos(alpha)])
        X  = np.array([Mx,My]) + a*np.cos(theta)*u +(b*np.sin(theta))*v
        calc = ((X[0]-x[i])**2)+((X[1]-y[i])**2) + calc
    if len(x)!=0:
        return calc/len(x)
    else:
        return calc

def circ(x,y,a,Mx,My):
    calc = 0
    for i in range(0,len(x)):
        R = np.linalg.norm(np.array([x[i]-Mx,y[i]-My]))
        calc = calc+(R-a)**2
    if len(x)!=0:
        return calc/len(x)
    else:
        return calc

def get_ellipse(rim_coor,plot_on=False,remove_outlayers=False,shape='ellipse',iterations=1,step=2):
    #rim_coor = np.array(rim_coor)
     

    print(len(rim_coor),end="\r")


    ### remove_outlayers
    func = lambda params: get_middle(rim_coor,params)
    guess = [np.mean(rim_coor[:,0]),np.mean(rim_coor[:,1])]
    result = scipy.optimize.minimize(func,guess)
    M0 = result.x
    
    vec_list = []
    phi_list = []
    r_list = []
    for i in range(0,len(rim_coor)):
        vec_list.append(rim_coor[i][0:2]-M0)
        phi_list.append(np.arctan(vec_list[-1][1]/vec_list[-1][0]))
        r_list.append(np.linalg.norm(vec_list[-1]))

    mu = np.mean(r_list)
    std = np.std(r_list)

    if remove_outlayers==True:
        r_list_sel=[]
        vec_list_sel = []
        phi_list_sel = []
        rim_coor_sel = []
        for i in range(0,len(r_list)):
            if r_list[i]<=mu+2*std and r_list[i]>=mu-2*std:
                r_list_sel.append(r_list[i])
                vec_list_sel.append(vec_list[i])
                phi_list_sel.append(phi_list[i])
                rim_coor_sel.append(rim_coor[i])
    else:
        r_list_sel = r_list
        vec_list_sel = vec_list
        phi_list_sel = phi_list
        rim_coor_sel = rim_coor

    rim_coor_sel = np.array(rim_coor_sel)

    if plot_on==True:
        plt.plot(rim_coor[:,0],rim_coor[:,1],'.',label='Original')
        plt.plot(rim_coor_sel[:,0],rim_coor_sel[:,1],'.',label='After removed outlayers')
        #plt.plot(phi_list,r_list,'.',label='Original')
        #plt.plot(phi_list_sel,r_list_sel,'.',label='Removed outlayers')
        plt.legend(loc='best')
    
    if shape=='fill':
        AREA = fill_rim(A_ring,rim_coor_sel[:,0],rim_coor_sel[:,1])
        area = np.nansum(AREA)
        return [np.nan,np.nan,np.nan,np.nan], area, np.nan, np.nan,rim_coor_sel[:,0],rim_coor_sel[:,1],np.nan, np.nan, np.nan, AREA
    else:
        # Fit ellipse parameters
        func = lambda params: get_middle(rim_coor_sel,params)
        guess = [np.mean(rim_coor_sel[:,0]),np.mean(rim_coor_sel[:,1])]
        result = scipy.optimize.minimize(func,guess)
        M_new = result.x

        x = rim_coor_sel[:,0]
        y = rim_coor_sel[:,1]

        def ellipse_abalpha(x,y,M,params):
            [a,b,alpha] = params
            return ellipse(x,y,a,b,alpha,M[0],M[1])

        def ellipse_alpha(x,y,a,b,M,params):
            [alpha] = params
            return ellipse(x,y,a,b,alpha,M[0],M[1])

        def ellipse_ab(x,y,alpha,M,params):
            [a,b] = params
            return ellipse(x,y,a,b,alpha,M[0],M[1])

        def ellipse_M(x,y,a,b,alpha,params):
            [Mx,My] = params
            return ellipse(x,y,a,b,alpha,Mx,My)
        
        def circ_a(x,y,M,params):
            [a] = params
            return circ(x,y,a,M[0],M[1])

        def circ_M(x,y,a,params):
            [Mx,My] = params
            return circ(x,y,a,Mx,My)

        a_fit = max([(max(x)-min(x))/2.0,(max(y)-min(y))/2.0])
        b_fit = min([(max(x)-min(x))/2.0,(max(y)-min(y))/2.0])
        alpha_fit = 0.0
        Mx_fit = M_new[0]
        My_fit = M_new[1]
        M_fit = M_new
        
        n=0
        success = []
        while n < iterations:
            if shape=='circ':
                func_circ1 = lambda params: circ_a(x,y,[Mx_fit,My_fit],params)
                guess1 = [a_fit]
                result = scipy.optimize.minimize(func_circ1,guess1)
                [a_fit] = result.x
                success.append(result.success)

                func_circ2 = lambda params: circ_M(x,y,a_fit,params)
                guess1 = [Mx_fit,My_fit]
                result = scipy.optimize.minimize(func_circ2,guess1)
                [Mx_fit,My_fit] = result.x
                success.append(result.success)
                b_fit = a_fit
                alpha_fit=0
                n=iterations-1

            elif shape=='ellipse':
                func_ellipse1 = lambda params: ellipse_alpha(x,y,a_fit,b_fit,[Mx_fit,My_fit],params)
                guess1 = [alpha_fit]
                result = scipy.optimize.minimize(func_ellipse1,guess1)
                [alpha_fit] = result.x
                success.append(result.success)
                
                func_ellipse2 = lambda params: ellipse_ab(x,y,alpha_fit,[Mx_fit,My_fit],params)
                guess2 = [a_fit,b_fit]
                result = scipy.optimize.minimize(func_ellipse2,guess2)
                [a_fit,b_fit] = result.x
                success.append(result.success)
                
                func_ellipse3 = lambda params: ellipse_M(x,y,a_fit,b_fit,alpha_fit,params)
                guess3 = [Mx_fit,My_fit]
                result = scipy.optimize.minimize(func_ellipse3,guess3)
                success.append(result.success)
                M_fit = result.x
                [Mx_fit,My_fit] = M_fit

            n = n+1

        ### Redo without outlyers:
        if remove_outlayers == True:
            r2_err = np.array([ellipse([x[i]],[y[i]],a_fit,b_fit,alpha_fit,Mx_fit,My_fit) for i in range(0,len(x))])
            check = (r2_err - np.median(r2_err))<=50
            x_old = x
            y_old = y
            x = []
            y = []

            for i in range(0,len(check)):
                if check[i] == True:
                    x.append(x_old[i])
                    y.append(y_old[i])

            x = np.array(x)
            y = np.array(y)
            
            n=0
            while n < iterations:
                if shape=='circ':
                    func_circ1 = lambda params: circ_a(x,y,[Mx_fit,My_fit],params)
                    guess1 = [a_fit]
                    result = scipy.optimize.minimize(func_circ1,guess1)
                    [a_fit] = result.x
                    success.append(result.success)

                    func_circ2 = lambda params: circ_M(x,y,a_fit,params)
                    guess1 = [Mx_fit,My_fit]
                    result = scipy.optimize.minimize(func_circ2,guess1)
                    [Mx_fit,My_fit] = result.x
                    M_fit = result.x
                    success.append(result.success)
                    b_fit = a_fit
                    alpha_fit=0

                elif shape=='ellipse':
                    func_ellipse1 = lambda params: ellipse_alpha(x,y,a_fit,b_fit,[Mx_fit,My_fit],params)
                    guess1 = [alpha_fit]
                    result = scipy.optimize.minimize(func_ellipse1,guess1)
                    [alpha_fit] = result.x

                    func_ellipse2 = lambda params: ellipse_ab(x,y,alpha_fit,[Mx_fit,My_fit],params)
                    guess2 = [a_fit,b_fit]
                    result = scipy.optimize.minimize(func_ellipse2,guess2)
                    [a_fit,b_fit] = result.x

                    func_ellipse3 = lambda params: ellipse_M(x,y,a_fit,b_fit,alpha_fit,params)
                    guess3 = [Mx_fit,My_fit]
                    result = scipy.optimize.minimize(func_ellipse3,guess3)
                    M_fit = result.x
                    [Mx_fit,My_fit] = M_fit
                n = n+1


            #func_ellipse1 = lambda params: ellipse_abalpha(x,y,[Mx_fit,My_fit],params)
            #guess1 = [a_fit,b_fit,alpha_fit]
            #result = scipy.optimize.minimize(func_ellipse1,guess1)
            #[a_fit,b_fit,alpha_fit] = result.x
            #func_ellipse2 = lambda params: ellipse_M(x,y,a_fit,b_fit,alpha_fit,params)
            #guess2 = [Mx_fit,My_fit]
            #result = scipy.optimize.minimize(func_ellipse2,guess2)
            #M_fit = result.x
            #[Mx_fit,My_fit] = M_fit
        else:
            r2_err = np.nan*x
        
        #calc = ellipse_calc(x,y,a_fit,b_fit,alpha_fit,Mx_fit,My_fit,steps=101)
        calc=0#...adjust

        x_fit=[] #...adjust: leave out for faster computatuon
        y_fit=[]
        theta_vec=np.linspace(0,2*np.pi,101)
        for theta in theta_vec:
            u = np.array([np.cos(alpha_fit),np.sin(alpha_fit)])
            v = np.array([-np.sin(alpha_fit),np.cos(alpha_fit)])
            X  = M_fit + a_fit*np.cos(theta)*u +(b_fit*np.sin(theta))*v
            x_fit.append(X[0])
            y_fit.append(X[1])
        x_fit = np.array(x_fit)
        y_fit = np.array(y_fit)
        
        if plot_on==True:
            f,ax = plt.subplots()
            ax.imshow(A_trace)
            ax.plot([M_fit[0]],[M_fit[1]],'bo')
            ax.plot(x_fit,y_fit,'r',linewidth=6,alpha=0.3)

        area = a_fit*b_fit*np.pi
        
        M_fit = np.array(M_fit)+step

        return [a_fit,b_fit,alpha_fit,M_fit], area, x_fit+step, y_fit+step,x+step,y+step,r2_err, success,calc, np.nan

def get_contourfit(contours,image,plot_on=False,distmin=10,remove_outlayers=False,shape='ellipse',iterations=5,step=0,n_sel=1,names=['a','b','alpha','M','area', 'xfit', 'yfit', 'xdata', 'ydata','r2', 'success','c_area','rim_coor'],fitrun=True,sort='area',area_min=10,border_margin=20,circularity_min=1.0,point_min=10,mergeconts=False):
    ret={}
    if len(contours)!=0:
        # Remove close to border
        h,w = np.shape(image)[0:2]
        contours1 = []
        for c in contours:
            if True in list(c[:,:,0]<border_margin) or True in list(c[:,:,0]>h-border_margin) or True in list(c[:,:,1]<border_margin) or True in list(c[:,:,1]>w-border_margin):
                pass
            else:
                contours1.append(c)
        
        # Remove small areas
        area = [cv2.contourArea(c) for c in contours1]
        contours2 = []
        for c in contours1:
            if cv2.contourArea(c)<area_min and len(c)<point_min:
                pass
            else:
                contours2.append(c)
         
        # remove lines
        contours3 = []
        area = []
        cent = []
        circularity = []
        for c in contours2:
            val = cv2.contourArea(c)/cv2.arcLength(c,False)
            #if val>circularity_min and len(c)>point_min:
            if len(c)>point_min:
                circularity.append(val)
                contours3.append(c)
                area.append(cv2.contourArea(c))
                M = cv2.moments(c)
                try:
                    cent.append([M["m10"] / M["m00"],M["m01"] / M["m00"]])
                except ZeroDivisionError as e:
                    cent.append(np.nan)
            else:
                #print(val,cv2.contourArea(c),len(c))
                pass
        circularity = np.array(circularity)
        area = np.array(area)
        cent = np.array(cent)
        
        if mergeconts==True:
            print('len: ',len(contours3))
            contours3 = merge_close_contours(contours3,image,dist_max=3)
        area = []
        cent = []
        circularity = []
        for c in contours3:
            circularity.append(cv2.contourArea(c)/cv2.arcLength(c,False))
            area.append(cv2.contourArea(c))
            M = cv2.moments(c)
            try:
                cent.append([M["m10"] / M["m00"],M["m01"] / M["m00"]])
            except ZeroDivisionError as e:
                cent.append(np.nan)
        circularity = np.array(circularity)
        area = np.array(area)
        cent = np.array(cent)


        # Sorting
        if len(contours3)>1:
            if sort=='circ':
                i_dist = [x for _, x in sorted(zip(circularity, np.arange(0,len(contours3))))]
                i_dist = np.flip(i_dist)
            elif sort=='area':
                i_dist = [x for _, x in sorted(zip(area, np.arange(0,len(contours3))))]
                i_dist = np.flip(i_dist)
            elif sort=='count':
                counts = np.array([len(c) for c in contours3])
                i_dist = [x for _, x in sorted(zip(counts, np.arange(0,len(contours3))))]
                i_dist = np.flip(i_dist)
            elif sort=='cent': # True center is center of largest area
                i0 = list(area).index(np.nanmax(area))
                dist = [((cent[i][0]-cent[i0][0])**2+(cent[i][1]-cent[i0][1])**2)**0.5 for i in range(0,len(contours3))]
                i_dist = [x for _, x in sorted(zip(dist, np.arange(0,len(contours3))))]
        else:
            i_dist = [0]
        
        if len(i_dist)>1:
            contours4 = [contours3[i_dist[i]] for i in range(0,len(i_dist))]
            area_new = [area[i_dist[i]] for i in range(0,len(i_dist))]
            circularity_new = [circularity[i_dist[i]] for i in range(0,len(i_dist))]
            cent_new = [cent[i_dist[i]] for i in range(0,len(i_dist))]
            area = np.array(area_new)
            circ = np.array(circularity_new)
            cent = np.array(cent_new)
        else:
            contours4 = contours3

        c_sel = contours4

        if plot_on==True:
            f,ax = plt.subplots(2,1,figsize=(20,10))
            #ax[0].imshow(image)
            ax[1].imshow(image)

        for c in c_sel:
            if type(c)==float:
                rim_coor = np.nan
                c_area = np.nan
                c_M = np.nan
                c_moments = np.nan

            else:
                rim_coor = []
                for i in range(0,len(c)):
                    i,j = c[i][0]
                    rim_coor.append([i,j,image[j,i]])
                rim_coor = np.array(rim_coor)

                c_area = cv2.contourArea(c)
                M = cv2.moments(c)
                try:
                    c_M = [M["m10"] / M["m00"],M["m01"] / M["m00"]]
                except:
                    c_M = [np.inf,np.inf]
                c_moments = M

            if fitrun==True and np.isnan(c)==False:
                [a,b,alpha,M], area, xfit, yfit,xdata,ydata,r2, success,calc, z = get_ellipse(rim_coor,plot_on=plot_on,remove_outlayers=remove_outlayers,shape=shape,iterations=iterations,step=step)
            else:
                a = np.nan
                b = np.nan
                alpha=np.nan
                M = np.nan
                area = np.nan
                xfit = np.nan
                yfit = np.nan
                xdata = np.nan
                ydata = np.nan
                r2 = np.nan
                success = np.nan
                calc = np.nan
                z = np.nan

            for n in names:
                try:
                    ret[n]
                except:
                    ret[n] = []
                ret[n].append(locals()[n])


        if plot_on==True:
            ax[0].plot(x,y,alpha=0.5)
            ax[1].plot(x_fit,y_fit,alpha=0.5)

    for n in names:
        try:
            ret[n]
        except:
            ret[n] = [np.nan]*n_sel


        if plot_on==True:
            ax[0].plot(x,y,alpha=0.5)
            ax[1].plot(x_fit,y_fit,alpha=0.5)

    return ret

def auto_brightandcontrast(input_img, channel, clip_percent=1,contrast=1.0,brightness=1.0):
    #print('Brightness: ',brightness)

    if brightness==-1:
        return cv2.medianBlur(input_img,5)
    else:
        try:
            # https://stackoverflow.com/questions/56388949/i-want-to-increase-brightness-and-contrast-of-images-in-dynamic-way-so-that-the
            histSize=180
            alpha=0
            beta=0
            minGray=0
            maxGray=0
            accumulator=[]

            if(clip_percent==0):
                #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
                return input_img

            else:
                hist = cv2.calcHist([input_img],[channel],None,[256],[0, 256])
                accumulator.insert(0,hist[0])    

                for i in range(1,histSize):
                    accumulator.insert(i,accumulator[i-1]+hist[i])

                maxx=accumulator[histSize-1]
                minGray=0

                clip_percent=clip_percent*(maxx/100.0)
                clip_percent=clip_percent/2.0

                while(accumulator[minGray]<clip_percent[0]):
                    minGray=minGray+1

                maxGray=histSize-1
                while(accumulator[maxGray]>=(maxx-clip_percent[0])):
                    maxGray=maxGray-1

                inputRange=maxGray-minGray

                alpha=(histSize-1)/inputRange
                beta=-minGray*alpha

                out_img=input_img.copy()

                cv2.convertScaleAbs(input_img,out_img,alpha,beta)
                
                adjusted = cv2.convertScaleAbs(out_img, alpha=contrast, beta=brightness)
                return adjusted
        except Exception as e:
            plt.imshow(input_img)
            raise ValueError(str(e))

def remove_under_contour(image,c,val='Default',step=3,inside=True,mode='circ'):
    mask = np.zeros(image.shape,dtype=np.uint8)
    try:
        if val!='Default':
            md1 = val[0]
            md2 = val[1]
            md3 = val[2]
    except:
        pass
    if val=='Default':
        md1 = np.nanmedian(image[:,:,0])
        md2 = np.nanmedian(image[:,:,1])
        md3 = np.nanmedian(image[:,:,2])
    
    im = image.copy()
    #im = np.array(im,dtype=int)
    imval = image.copy()
    imval = np.array(imval,dtype=int)
    
    if mode=='ellipse':
        x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res, res_list = ELPS.run_ellipse(c,scale=1.0)

    print(mode,end='\r')
    (x,y),r = cv2.minEnclosingCircle(c)

    coor = []
    valnew = []

    cx = c[:,:,1].flatten()
    cy = c[:,:,0].flatten()
    rim_coor = np.array([[cx[i],cy[i]] for i in range(0,len(cx))])
    for i in range(0,len(image)):
        for j in range(0,len(image[i])):
            if mode=='ellipse':
                remove=0
                select=False
                if inside==False:
                    inout = (((j-x0)/(ap+step))**2)+(((i-y0)/(bp+step))**2)
                    inout2 = (((j-x0)/(ap+2*step))**2)+(((i-y0)/(bp+2*step))**2)
                    if inout<=1:
                        remove=1
                    elif inout>1 and inout2<=1:
                        select = True
                elif inside==True:
                    inout = (((j-x0)/(ap-step))**2)+(((i-y0)/(bp-step))**2)
                    inout2 = (((j-x0)/(ap-2*step))**2)+(((i-y0)/(bp-2*step))**2)
                    if inout>=1:
                        remove=1
                    elif inout<1 and inout2>=0:
                        select = True
                if remove==1:
                    coor.append([i,j])
                    mask[i,j] = (255,255,255)
                if select==True:
                    valnew.append(im[i,j])


            elif mode=='circ':
                if inside==False:
                    #val = np.nanmin((cy-i)**2+(cx-j)**2)
                    if ((y-i)**2+(x-j)**2)<(r-step)**2:
                        #if val<(r-step)**2:
                        #im[i,j,0] = np.nan
                        #im[i,j,1] = np.nan
                        #im[i,j,2] = np.nan
                        #im[i,j] = np.array([np.nan, np.nan, np.nan]) 
                        coor.append([i,j])

                        mask[i,j] = (255,255,255)
                    
                    if (((y-i)**2+(x-j)**2)>(r+step)**2) and (((y-i)**2+(x-j)**2)<=(r+step+2)**2):
                        valnew.append(im[i,j])

                elif inside==True:
                    val = np.nanmin((cy-i)**2+(cx-j)**2)**0.5
                    #if ((y-i)**2+(x-j)**2)>(r-step)**2:
                    if val<step:
                        #im[i,j] = np.array([np.nan, np.nan, np.nan]) 
                        #im[i,j,0] = np.nan
                        #im[i,j,1] = np.nan
                        #im[i,j,2] = np.nan
                        coor.append([i,j])
                        
                        mask[i,j] = (255,255,255)
                    
                    if (((y-i)**2+(x-j)**2)<(r+step)**2) and (((y-i)**2+(x-j)**2)>=(r+step-2)**2):
                        valnew.append(im[i,j])
    
    val = np.nanmedian(valnew,axis=0)
    #print(type(val))

    #    print(valnew)
    for c in coor:
        im[c[0],c[1]] = val
    

    return im, mask

def get_all_contours(image,operator,th1=60,th2=40,center_off=10,step=5,names=['a', 'b', 'alpha', 'M', 'area', 'xfit', 'yfit', 'xdata', 'ydata','r2', 'success', 'calc', 'rim_coor', 'c_area', 'c_M', 'c_moments','c'],fitrun=False,inside=True,inside_try=False,edge_val=0.5,origin_val=1.0,pixmax=False,circularity_min=0.0,split_contours=False,point_min=10,contrast=1.0,brightness=1.0,mergeconts=False,clip_percent=1,mode='Grids',ransac_opt=None):
    sortc = 'area'
    if mode=='Grids':
        modescale = 1.0
    elif mode=='Holes':
        modescale = 5.0

    out = []
    edges_all = []
    run=True
    # Find first rim
    try:
        q1 = auto_brightandcontrast(image, 0, clip_percent=clip_percent,contrast=contrast,brightness=brightness)
        #plt.figure()
        #plt.imshow(q1)
        #plt.show()
    except ValueError as e:
        print(e)
        q1 = image
        print('Did not enhane contrast')
    
    img_grey1, contours, hierarchy,edges,pixmax = get_edges(q1,operator,thresh=th1,a=edge_val,b=origin_val,pixmax=pixmax,clip_percent=clip_percent)
    ret = get_contourfit(contours,image,plot_on=False,distmin=center_off,remove_outlayers=False,shape='ellipse',iterations=5,step=0,n_sel=100,names=names,fitrun=fitrun,sort=sortc,circularity_min=circularity_min,point_min=point_min,mergeconts=mergeconts)
    edges_all.append(edges)
    IMG_ENH.append(q1)
    #plt.figure()
    #plt.imshow(edges)
    #plt.imshow(q1)
    #for c in ret['c']:
    #    plt.plot(c[:,:,0],c[:,:,1])
    #plt.show()
    
    M0 = np.array(ret['c_M'][0])
    m0 = np.array(np.shape(image)[0:2])*0.5
    print(np.linalg.norm(M0-m0))
    if np.linalg.norm(M0-m0)<center_off:
        for i in range(0,len(ret['c'])):
            if np.linalg.norm(np.array(ret['c_M'][i])-M0)<center_off:
                out.append(ret['c'][i])
    else:

        print('Did not find proper contours')
        print(np.linalg.norm(M0-m0)<center_off)
        
    #plt.figure()
    #plt.imshow(image)
    #for c in out:
    #    plt.plot(c[:,:,0],c[:,:,1])
    
    if len(out)>=1: #...adjusted
        run=True
        if run==True:
            # Mask inner crater
            cent, area, M, circ = get_c_params(out)
            imax = area.index(np.nanmax(area))
            imin = area.index(np.nanmin(area))
            M = cent[imax]
            M0 = cent[imin]
            c = out[imax]
            if len(out)>1:
                if cv2.contourArea(out[1])/cv2.contourArea(out[0])>0.7:
                    c = out[1]
                else:
                    c = out[0]
            else:
                c = out[0]
            
            found = False
            if inside_try==True:
                inside_list = [inside,bool(not inside)]
            elif inside_try==False:
                inside_list = [inside]

            q_all = []
            mask_all = []
            im_out_all = []
            brightcheck = []
            for inside_sel in inside_list:
                q,mask = remove_under_contour(image,c,step=step,inside=inside_sel,mode='ellipse')
                im_out = q
                
                q_all.append(q)
                mask_all.append(mask)
                im_out_all.append(im_out)

                qq = image*np.where(mask==(255,255,255),np.nan,1)
                brightcheck.append(np.nanmean(operator(qq)))
            
            if len(inside_list)==1:
                inside_sel = 0
            else:
                inside_sel = brightcheck.index(np.nanmin(brightcheck))
            q = q_all[inside_sel]
            mask = mask_all[inside_sel]
            im_out = im_out_all[inside_sel]
            inside_sel = inside_list[inside_sel]
            print('inside sel is '+str(inside_sel))

            #Remove spots with different crater center
            try:
                q2 = auto_brightandcontrast(im_out, 0, clip_percent=clip_percent,contrast=contrast,brightness=brightness)
                img_grey1, contours, hierarchy,edges,thresh_image, pixmax = get_edges(q2,operator,thresh_output=True,a=edge_val,b=origin_val,pixmax=False)
                ret2 = get_contourfit(contours,image,plot_on=False,distmin=100,remove_outlayers=False,shape='ellipse',iterations=5,step=0,n_sel=20,names=names,fitrun=fitrun,sort=sortc,circularity_min=circularity_min,point_min=point_min,mergeconts=mergeconts)
                found=True
                edges_all.append(edges)
                IMG_ENH.append(q2)
            except Exception as e:
                ret2={'c':[]}
                print(e)
                found=False
             
            if inside_sel==False:
                center_off = center_off*3
            
            if found==True:
                if 'float' not in str(type(ret2['c'][0])):
                    # Merge contours
                    q3 = np.copy(thresh_image)

                    for c in contours:
                        cv2.fillPoly(q3, pts=[c], color=0)
                    
                    for c in ret2['c']:
                        if cv2.contourArea(c)>100:
                            cv2.fillPoly(q3, pts=[c], color=255)
                    
                    binr = q3
                    
                    # define the kernel
                    kernel = np.ones((3, 3), np.uint8)

                    # dilate the image
                    dilation = cv2.dilate(binr, kernel, iterations=1)

                    dilcont, dilhier = cv2.findContours(q3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    c_merge = []
                    i_merge = []
                    for i in range(0,len(dilcont)):
                        c_merge.append([])
                        for j in range(0,len(ret2['c'])):
                            c = ret2['c'][j]
                            if cv2.pointPolygonTest(dilcont[i], (int(c[0][0][0]),int(c[0][0][1])), False)>=0:
                                c_merge[-1].append(c)
                                i_merge.append(j)

                    for i in range(0,len(ret2['c'])):
                        if i not in i_merge:
                            c_merge.append([ret2['c'][i]])
                    
                    cout = []
                    for c in c_merge:
                        if len(c)!=0:
                            cnew = merge_conts(c)
                        else:
                            cnew = c
                        cout.append(cnew)
                    
                    ret3 = {}
                    ret3['c'] = cout

            if 'ret3' not in locals():
                ret3 = {'c':[]}
 
            # Fit ellipses
            try:
                x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1  = ELPS.run_ellipse_rs(out[0],scale=1.0,ransac_opt=ransac_opt)
                ref = np.array([x01,y01,ap1,bp1])
            except Exception as e:
                plt.imshow(image)
                plt.plot(out[0][:,:,0],out[0][:,:,1])
                plt.show()
            for c in ret3['c'][0:30]:
                try:
                    fitparam = ELPS.run_ellipse_rs(c,scale=1.0,ransac_opt=ransac_opt)
                    if (np.linalg.norm(fitparam[4][0:2]-ref[0:2]))<center_off:
                        val = np.array(fitparam[4][2:4])
                        if (np.nanmax(np.abs((val-ref[2:4])/ref[2:4]))< 0.5 and inside_sel==True) or (np.nanmax(np.abs((val-ref[2:4])/ref[2:4]))> 1.2 and inside_sel==False):
                            out.append(c)
                except Exception as e:
                    print(e,end='\r')
            
            if split_contours==True:
                out_new = []
                for c in out:
                    if len(c)>20:
                        try:
                            c_split, state = split_double_rim(c,lim=2,ransac_opt=ransac_opt)
                        except Exception as e:
                            print(str(e),end='\r')
                            c_split = []
                        for c_new in c_split:
                            out_new.append(c_new)

                out = out_new

            area = []
            for c in out:
                area.append(cv2.contourArea(c))
            try:
                out_sort = np.array([x for _, x in sorted(zip(area, out))])
            except Exception as e:
                order = list(range(0,len(out)))
                order_sort = np.array([x for _, x in sorted(zip(area, order))])
                out_sort = [out[i] for i in order_sort]
                #raise ValueError(str(e))
            if len(out_sort)>1:
                out_sort = np.flip(out_sort)
    
    if len(out)<=1:
        out_sort = out
    for val in edges_all:
        EDGES.append(val)
    return out_sort, pixmax, edges_all[0] #, q1 #, test#, [image,q1,im_out,im_out2,q2,q3]

def shift_by_contour(images,pix_size,i_vec,j_vec,operator,thresh,gridsize=[6,6],loc='ARCNL',pixmax=False,a=1.0,b=0.0,contrast=1.0,lowcontrast=False,sel=-1,clip_percent=1):
    image = images[sel]
    # Find first rim
    q1 = auto_brightandcontrast(image, 0, clip_percent=clip_percent,contrast=1.0)
    img_grey1, contours, hierarchy,edges,pixmax = get_edges(q1,operator,thresh=thresh,pixmax=pixmax,a=a,b=b)
    ret = get_contourfit(contours,image,plot_on=False,distmin=10,remove_outlayers=False,shape='ellipse',iterations=5,step=0,n_sel=20,names=['c'],fitrun=False,sort='area',circularity_min=0.0)
    
    if lowcontrast==True:
        # Remove close to border
        border_margin=10
        point_min=10
        area_min=10
        h,w = np.shape(image)[0:2]
        contours1 = []
        for c in contours:
            if True in list(c[:,:,0]<border_margin) or True in list(c[:,:,0]>h-border_margin) or True in list(c[:,:,1]<border_margin) or True in list(c[:,:,1]>w-border_margin):
                pass
            else:
                contours1.append(c)

        # Remove small areas
        area = [cv2.contourArea(c) for c in contours1]
        contours2 = []
        for c in contours1:
            if cv2.contourArea(c)<area_min and len(c)<point_min:
                pass
            else:
                contours2.append(c)
        area = [cv2.contourArea(c) for c in contours2]
        area = [len(c) for c in contours2]
        csel = contours2[area.index(np.nanmax(area))]
        j_off,i_off = get_c_params([csel])[0][0] #..adjusted. Before: ret['c'][0:1]

    elif lowcontrast==False:
        img_grey_sel, contours, hierarchy,edges, pixmax = get_edges(image,operator,thresh=60,thresh_output=False,a=1.0,b=0.0,sigma=4.0,pixmax=False,lt=0.0,ht=0.2)
        x = np.array([])
        y = np.array([])
        for c in contours:
            #plt.plot(c[:,:,0],c[:,:,1])
            x = np.concatenate((x,c[:,:,0].flatten()))
            y = np.concatenate((y,c[:,:,1].flatten()))

        x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res, res_list = ELPS.run_ellipse_rs([x,y],scale=1.0)
        j_off,i_off = x0,y0

    #j_off,i_off = get_c_params([csel])[0][0] #..adjusted. Before: ret['c'][0:1]
    di = (200.0e-6/pix_size)
    dj = di

    i_end = i_vec[-1]-0.5*np.shape(image)[0]+i_off
    j_end = j_vec[-1]-0.5*np.shape(image)[1]+j_off

    i_new = []
    j_new = []
    for i in range(-gridsize[0]+1,1):
        i_new.append(i*di+i_end)
    if loc=='ARCNL':
        for j in range(gridsize[1]-1,-1,-1):
            j_new.append(j*dj+j_end)
    elif loc=='UU':
        for j in range(gridsize[1]+1,1):
            j_new.append(j*dj+j_end)

    i_new = np.array(i_new)
    j_new = np.array(j_new)

    return i_new,j_new

def run_twice(image,operator,th1,th2,center_off,step=2,n=4,names=[],edge_val=0.5,origin_val=1.0,inside=True,inside_try=False,pixmax=False,point_min=10,contrast=3.0,brightness=1.0,enhanced_contrast=True,clip_percent=1,mode='Grids',ransac_opt=None):
    pixmax0 = pixmax
    C_all = []

    for pixmax in [pixmax0]:#,1000000*0.001]:
        brightness_start=1.0
        contrast_start=1.0
        mergeconts = False
        if brightness!=-1.0:
            out1, pixmax1, edges1 = get_all_contours(image,operator,th1=th1,th2=th2,center_off=center_off,step=step,names=names,edge_val=edge_val,origin_val=origin_val,inside=inside,inside_try=inside_try,pixmax=pixmax,point_min=point_min,contrast=contrast_start,brightness=brightness_start,clip_percent=clip_percent,mode=mode,ransac_opt=ransac_opt)[0:n]
            print('pixmax: ',pixmax1)
        elif brightness==-1.0:
            out1 = []
            #mergeconts=True
        if enhanced_contrast==True:
            out2,pixmax2, edges2 = get_all_contours(image,operator,th1=th1,th2=th2,center_off=center_off,step=step,names=names,edge_val=edge_val,origin_val=origin_val,inside=inside,inside_try=inside_try,pixmax=pixmax,point_min=point_min,contrast=contrast,brightness=brightness,mergeconts=mergeconts,clip_percent=clip_percent,mode=mode,ransac_opt=ransac_opt)[0:n]
            if len(out2)!=0 and len(out1)!=0:
                if cv2.contourArea(out2[-1])<cv2.contourArea(out1[-1]):
                    out2 = out2[:-1]
        else:
            out2 = []

        out = list(out1)
        origin=[0]*len(out1)
        for OUT in out2:
            out.append(OUT)
            origin.append(1)
        origin = np.array(origin)
        #plt.figure()
        #plt.imshow(image)
        #for c in out2:
        #    plt.plot(c[:,:,0],c[:,:,1])
        #plt.show()
                
        if len(out)>0:
            #plt.figure()
            #plt.imshow(image)
            #for c in out:
            #    plt.plot(c[:,:,0],c[:,:,1])
            #plt.show()

            out_all = [list(out1),list(out2)]

            # Fit ellispes and return values
            C_new = []
            alpha_fit = []
            M_fit = []
            a_fit = []
            b_fit = []
            area_fit = []
            px_fit = []
            py_fit = []
            res_fit = []
            res_list_fit = []
            selout = []
            
            for s1 in range(0,len(out_all)):
                for s2 in range(0,len(out_all[s1])):
                    c = out_all[s1][s2]
                    selout.append([s1,s2])
                    state=True
                    if type(c)!=float:
                        try:
                            try:
                                x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res, res_list = ELPS.run_ellipse_rs(c,scale=1.0,ransac_opt=ransac_opt)
                            except Exception as e:
                                #print('rs')
                                #print(e)
                                pass
                            M = np.array([x0,y0])
                            a,b = np.array([ap,bp])
                            alpha = phi
                            E = res
                        except Exception as e:
                            #print(e,end='\r')
                            print(e)
                            state=False
                    elif type(c)==float or state==False:
                        M = np.array([np.nan,np.nan])
                        a = np.nan
                        b = np.nan
                        alpha = np.nan
                        e = ((np.nan,np.nan),(np.nan,np.nan),np.nan)
                        E = [[np.nan],[np.nan]]
                        xfit = [np.nan]
                        yfit = [np.nan]
                        res = np.inf
                        res_list = [np.inf]
                    alpha_fit.append(alpha)
                    a_fit.append(a)
                    b_fit.append(b)
                    M_fit.append(M)
                    area_fit.append(a*b*np.pi)
                    px_fit.append(xfit)
                    py_fit.append(yfit)
                    res_fit.append(res)
                    res_list_fit.append(res_list)

            # Filter non centric ellipses and off-center ellipses
            distmin = 10
            if mode== 'Holes':
                distmin = distmin*1.0
            i_areamax = area_fit.index(np.nanmax(area_fit))
            M0 = M_fit[i_areamax]
            outnew = []
            M_new = []
            area_new = []
            a_new = []
            b_new = []
            alpha_new = []
            px_new = []
            py_new = []
            res_new = []
            res_list_new = []
            origin_new = []
            selout_new = []
            for i in range(0,len(M_fit)):
                dist = (np.linalg.norm(M_fit[i]-M0))
                circ = np.nanmax([a_fit[i],b_fit[i]])/np.nanmin([a_fit[i],b_fit[i]])
                if dist<=distmin:
                    j = selout[i]
                    outnew.append(out_all[j[0]][j[1]])
                    area_new.append(area_fit[i])
                    M_new.append(M_fit[i])
                    a_new.append(a_fit[i])
                    b_new.append(b_fit[i])
                    alpha_new.append(alpha_fit[i])
                    selout_new.append(selout[i])
                    res_new.append(res_fit[i])
                    res_list_new.append(res_list_fit[i])
                    px_new.append(px_fit[i])
                    py_new.append(py_fit[i])
                    origin_new.append(origin[i])

            #plt.figure()
            #plt.imshow(image)
            #for c in outnew:
            #    plt.plot(c[:,:,0],c[:,:,1])
            #plt.show()

            area_fit = area_new
            M_fit = M_new
            a_fit = a_new
            b_fit = b_new
            alpha_fit = alpha_new
            px_fit = px_new
            py_fit = py_new
            res_fit = res_new
            res_list_fit = res_list_new
            origin = origin_new
            selout = selout_new

            # Group by area
            sortarea_lim = 0.1
            if mode=='Holes':
                sortarea_lim = sortarea_lim*5
            dict_sort = {}
            i_save = []
            for i in range(0,len(area_fit)):
                for j in range(i+1,len(area_fit)):
                    if i!=j:
                        val1 = area_fit[i]
                        val2 = area_fit[j]
                        val = np.abs((val1-val2)/np.nanmin([val1,val2]))
                        if val<=sortarea_lim:
                            key = i
                            for k in dict_sort.keys():
                                if (i in dict_sort[k]) or (j in dict_sort[k]):
                                    key = k
                            if key not in dict_sort.keys():
                                dict_sort[key] = [i,j]
                            else:
                                if i not in dict_sort[key]:
                                    dict_sort[key].append(i)
                                if j not in dict_sort[key]:
                                    dict_sort[key].append(j)
                            i_save.append(i)
                            i_save.append(j)
            
            for i in range(0,len(area_fit)):
                if i not in i_save:
                    dict_sort[i] = [i]
            
            origin_new = []
            outnew2 = []
            sortnew2 = []
            sel = []
            
            C = []
            for k in dict_sort.keys():
                conts = []
                ids = dict_sort[k]
                i_all = []
                j_all = []
                ij_all = []
                origin_new.append(origin[k])
                for idx in ids:
                    i,j = selout[idx]
                    i_all.append(i)
                    j_all.append(j)
                    ij_all.append([i,j])
                if 1 not in i_all:
                    for ij in ij_all:
                        conts.append(out_all[ij[0]][ij[1]])
                elif 1 in i_all:
                    for ij in ij_all:
                        if ij[0]==1:
                            conts.append(out_all[ij[0]][ij[1]])

                if len(conts)==1:
                    C.append(conts[0])
                else:
                    C.append(merge_conts(conts))
            origin = origin_new
            #plt.figure()
            #plt.imshow(image)
            #for c in C:
            #    plt.plot(c[:,:,0],c[:,:,1])
            #plt.show()
        else:
            img_grey_sel, contours, hierarchy,edges, pixmax = get_edges(image,operator,thresh=th1,thresh_output=False,a=edge_val,b=origin_val,sigma=4.0,pixmax=False,lt=0.0,ht=0.2)
            x = np.array([])
            y = np.array([])
            contours2 = []
            origin = []
            for c in contours:
                if len(c[:,:,0])>point_min:
                    contours2.append(c)
                    x = np.concatenate((x,c[:,:,0].flatten()))
                    y = np.concatenate((y,c[:,:,1].flatten()))
                    origin.append(0)

            try:
                x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res,res_list = ELPS.run_ellipse_rs([x,y],scale=1.0,ransac_opt=ransac_opt)
                cfind, status = split_double_rim([x,y],lim=2,fit=False,fitval=[x0,y0,ap,bp,e,phi])
                if status==True:
                    if len(cfind[0][:,:,0])>point_min and len(cfind[1][:,:,1])>point_min:
                        [cout,cin] = cfind 
                        C = cfind
                        origin = np.array([0]*len(C))
                        print('Did find contour')
                    else:
                        status=False
                if status==False:
                    C = []#cfind
                    origin = []
                    print('Did not find contour')
                    #plt.figure()
                    #plt.imshow(image)
                    #for c in out:
                    #    plt.plot(c[:,:,0],c[:,:,1])
                    #plt.plot(xfit,yfit,'r')
                    #plt.show()
            except Exception as e:
                print('No contours')
                print(str(e)) 
                C = []
                origin = []

        C_all.append(C)
    #l = [len(C) for C in C_all]
    #C = C_all[l.indx(max(l))]
    C = []
    for i in range(0,len(C_all)):
        for cont in C_all[i]:
            C.append(cont)
    return C,origin


def run_contours(images,th1,th2,center_off,operator,n=3,step=2,names=['a', 'b', 'alpha', 'M', 'area', 'xfit', 'yfit', 'xdata', 'ydata','r2', 'success', 'calc', 'rim_coor', 'c_area', 'c_M', 'c_moments','c'],names2=['a_fit','b_fit','alpha_fit','M_fit','cont','c_cent','c_area','c_M','c_circ','area_fit','C'],edge_val=0.5,origin_val=1.0,inside=True,inside_try=False,pixmax=False,point_min=10,contrast=3.0,brightness=1.0,enhanced_contrast=True,clip_percent=1,mode='Grids',ransac_opt=None,blur_func=lambda im: im,contr_bright_startfunc=lambda im: im):
    global EDGES,IMG_ENH
    EDGES = []
    IMG_ENH = []

    ### Bluring and remove Moire pattern (for SEM images)
    images_new = [contr_bright_startfunc(blur_func(im)) for im in images]
    images = images_new

    out = []
    origin = []
    if mode=='Holes':
        modescale = 1.0
        center_off = center_off*modescale*5
        step = step*modescale
        point_min = point_min*modescale
    # Get contours
    for idx in range(0,len(images)):
        OUT,ori = run_twice(images[idx],operator,th1,th2,center_off,step=step,n=n,names=names,edge_val=edge_val,origin_val=origin_val,inside=inside,inside_try=inside_try,pixmax=pixmax,point_min=point_min,contrast=contrast,brightness=brightness,enhanced_contrast=enhanced_contrast,clip_percent=clip_percent,mode=mode,ransac_opt=None)
        out.append(OUT)
        origin.append(ori)
        print(idx)

    # Select proper contours
    C = []
    ORI = []
    for i in range(0,len(out)):
        C.append([])
        ORI.append([])
        for j in range(0,n):
            try:
                C[i].append(out[i][j])
                ORI[i].append(origin[i][j])
            except:
                C[i].append(np.nan)
                ORI[i].append(np.nan)

    C = np.array(C)

    fitcv = {}
    for name in names2:
        fitcv[name] = []

    # Fit ellispes and return values
    C_new = []
    for idx in range(0,len(C)):
        C_new.append([])
        alpha_fit = []
        M_fit = []
        a_fit = []
        b_fit = []
        area_fit = []
        x_fit = []
        y_fit = []
        res_fit = []
        res_list_fit = []
        e_fit = []
        xdata = []
        ydata = []
        run = []

        for c in C[idx]:
            state=True
            if type(c)!=float:
                C_new[-1].append(c)
                try:
                    x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res,res_list = ELPS.run_ellipse_rs(c,scale=1.0,ransac_opt=ransac_opt)
                    #e = cv2.fitEllipse(c)
                    M = np.array([x0,y0])
                    a,b = np.array([ap,bp])
                    alpha = phi
                    E = res
                    #ax.plot(c[:,:,0],c[:,:,1],'--')
                    #plt.plot(E[:,0],E[:,1],color=ax.get_lines()[-1]._color)
                except Exception as err:
                    print(err,end='\r')
                    #print(err)
                    state=False
            elif str(type(c))==float or state==False or np.isnan(c):
                M = np.array([np.nan,np.nan])
                a = np.nan
                b = np.nan
                alpha = np.nan
                e = np.nan
                E = [[np.nan],[np.nan]]
                res = np.nan
                res_list = [np.nan]
                xfit = [np.nan]
                yfit = [np.nan]
                x = [np.nan]
                y = [np.nan]
            
            try:
                alpha
            except:
                M = np.array([np.nan,np.nan])
                a = np.nan
                b = np.nan
                alpha = np.nan
                e = np.nan
                E = [[np.nan],[np.nan]]
                res = np.nan
                res_list = [np.nan]
                xfit = [np.nan]
                yfit = [np.nan]
                x = [np.nan]
                y = [np.nan]

            alpha_fit.append(alpha)
            a_fit.append(a)
            b_fit.append(b)
            M_fit.append(M)
            area_fit.append(a*b*np.pi)
            x_fit.append(xfit)
            y_fit.append(yfit)
            e_fit.append(e)
            res_fit.append(res)
            res_list_fit.append(res_list)
            xdata.append(x)
            ydata.append(y)

        run = np.array(ORI[idx])

        cont = C_new[-1]
        c_cent, c_area, c_M, c_circ = get_c_params(C_new[-1])

        for name in names2:
            q = list(locals()[name])
            while len(q)<3:
                q.append(np.nan)
            fitcv[name].append(q)
    for k in fitcv.keys():
        fitcv[k] = np.array(fitcv[k])
    
    #plt.figure()
    #plt.imshow(EDGES[0])
    #plt.show()

    fitcv['EDGES'] = EDGES
    fitcv['IMG_ENH'] = IMG_ENH

    return fitcv

### Merge contours
# Source: https://stackoverflow.com/questions/44501723/how-to-merge-contours-in-opencv

class clockwise_angle_and_distance():
    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use: 
        instantiate with an origin, then call the instance during sort
    reference: 
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle
    
    distance
    

    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -np.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = np.arctan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*np.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        return angle, lenvector

def merge_conts(conts):
    if len(conts)>1:
        # Get order
        order = []
        for i in range(0,len(conts)):
            for j in range(0,len(conts)):
                if j!=i:
                    c1 = conts[i]
                    c2 = conts[j]

                    c1start = c1[0,0]
                    c1end = c1[-1,0]
                    c2start = c2[0,0]
                    c2end = c2[-1,0]

                    l1 = np.linalg.norm(c1start-c2start)
                    l2 = np.linalg.norm(c1start-c2end)
                    l3 = np.linalg.norm(c1end-c2start)
                    l4 = np.linalg.norm(c1end-c2end)
                    lmin = np.nanmin([l1,l2,l3,l4])
                    order.append([i,j,lmin])
        order = np.array(order)
        
        i_start = list(order[:,2]).index(np.nanmax(order[:,2]))
        isel, jsel = order[i_start][0:2]
        L = [jsel,isel]

        while len(L)<len(conts):
            dist = np.inf
            found=False
            for o in order:
                if (o[0]==L[-1] and o[1] not in L):
                    if o[2]<dist:
                        isel = o[1]
                        if isel not in L and isel!=jsel:
                            found=True
                            dist = o[2]

            if found==True:
                L.append(isel)
        
        L_new = L[1:]
        L_new.append(L[0])
        L_new = np.array(L_new,dtype=int)

        list_of_pts = []
        for i in range(1,len(L_new)):
            if i==1:
                c1 = conts[L_new[i-1]]
            else:
                c1 = c2
            c2 = conts[L_new[i]]

            c1start = c1[0,0]
            c1end = c1[-1,0]
            c2start = c2[0,0]
            c2end = c2[-1,0]

            l1 = np.linalg.norm(c1start-c2start)
            l2 = np.linalg.norm(c1start-c2end)
            l3 = np.linalg.norm(c1end-c2start)
            l4 = np.linalg.norm(c1end-c2end)
            lmin = np.nanmin([l1,l2,l3,l4])
            if l1 == lmin and i==1:
                c1 = np.flip(c1,axis=1)
            elif l2 ==lmin:
                if i==1:
                    c1 = np.flip(c1,axis=1)
                c2 = np.flip(c2,axis=1)
            elif l4==lmin:
                c2 = np.flip(c2,axis=1)

            if i==1:
                list_of_pts += [pt[0] for pt in c1]
            list_of_pts += [pt[0] for pt in c2]

        
        center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
        clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort
        ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)

        #area = [cv2.arcLength(c,False) for c in ctrs]
        #ctr = ctrs[area.index(np.nanmin(area))]
    else:
        ctr = conts[0]

    return ctr

def merge_close_contours(contours,image,dist_max=3,plot_on=False):
    idx_sort = {}
    idx_all = range(0,len(contours))
    idx_check = []
    for i in range(0,len(contours)):
        for j in range(0,len(contours)):
            if j!=i:
                c1 = contours[i]
                c2 = contours[j]
                x1 = c1[:,:,0].flatten()
                y1 = c1[:,:,1].flatten()
                x2 = c2[:,:,0].flatten()
                y2 = c2[:,:,1].flatten()
                p1 = np.transpose(np.array([x1,y1]))
                p2 = np.transpose(np.array([x2,y2]))
                val = []
                for p10 in p1:
                    for p20 in p2:
                        val.append(np.linalg.norm(p10-p20))
                dist = np.nanmin(val)

                if dist<=dist_max:
                    k0 = min(i,j)
                    if k0 not in idx_check:
                        k = k0
                        idx_sort[k] = []
                    else:
                        for k1 in idx_sort.keys():
                            if k0 in idx_sort[k1]:
                                k = k1
                    if i not in idx_sort[k]:
                        idx_sort[k].append(i)
                        idx_check.append(i)
                    if j not in idx_sort[k]:
                        idx_sort[k].append(j)
                        idx_check.append(j)
    
    for i in range(0,len(contours)):
        if i not in idx_check:
            idx_sort[i] = [i]
    
    c_merge = []
    for k in idx_sort.keys():
        c_merge.append([])
        for idx in idx_sort[k]:
            c_merge[-1].append(contours[idx])

    c_new = []
    for c in c_merge:
        c_new.append(merge_conts(c))
    
    if plot_on==True:
        f,ax = plt.subplots(1,2)
        ax[0].imshow(image)
        ax[1].imshow(image)
        for c in contours:
            ax[0].plot(c[:,:,0],c[:,:,1])
        for c in c_new:
            ax[1].plot(c[:,:,0],c[:,:,1])
        f.show()

    return c_new

def inspect_inside_contour(c,image,th=160,contrast=1.0,brightness=1.0,countlim=100,clip_percent=1):
    q1 = auto_brightandcontrast(image, 0, clip_percent=clip_percent,contrast=contrast,brightness=brightness)
    #p = []
    count = 0
    try:
        for i in range(0,len(q1)):
            for j in range(0,len(q1[i])):
                if cv2.pointPolygonTest(c, (i,j), False)>0:
                    val = np.nanmax(image[i,j])
                    #p.append([i,j,val])
                    if val>th:
                        count = count+1
                        if count==countlim:
                            raise StopIteration
    except StopIteration:
        pass

    #p = np.array(p)

    return count

def threshold_by_background(image,operator,ploton=False,a=2,steps=10000):
    # a = times the FWHM cutoff

    ### get only background image
    d = np.min(np.shape(image)[0:2])
    r = 0.8*d*0.5

    x = np.array(range(0,np.shape(image)[1]))
    y = np.array(range(0,np.shape(image)[0]))
    x1, y1 = np.meshgrid(x,y)
    mask = np.where(((x1-np.nanmean(x1))**2+(y1-np.nanmean(y1))**2)>=r**2,1,np.nan)
    MASK = np.copy(image)
    for i in range(0,len(image[0,0])):
        MASK[:,:,i] = mask

    #image = abs(image - np.nanmedian(image,axis=[0,1]))
    #image = np.array(image,dtype=np.uint8)
    im = operator(image)
    #plt.imshow(im*MASK[:,:,0]) 
    #plt.imshow(MASK[:,:,0]) 
    #im = np.array(abs(im-np.nanmedian(im)),dtype=np.uint8)
    
    # Calculate histogram
    vals = (im*mask).flatten()
    vals = vals[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    med = np.median(vals)
    #med = np.mean(vals)
    std = np.nanstd(vals)
    start = med-a*std
    if start<0:
        start=0
    stop = med+a*std
    nbins = steps
    hist, bins = np.histogram(vals,bins = steps,range=(start,stop))
    bins = (bins[1:]+bins[:-1])*0.5
    
    # Fit gaussion on background pixels
    f_gauss = lambda t,P0,t0,t_FWHM: P0*np.exp(-4*np.log(2)*(((t-t0)/t_FWHM)**2))
    mod_gauss = models.Model(f_gauss,name='gauss')
    params = mod_gauss.make_params()
    params['t0'].value = bins[list(hist).index(np.nanmax(hist))]
    params['t_FWHM'].value = std*2.0*((2.0*np.log(2))**0.5)
    params['t_FWHM'].min = 0.0
    params['P0'].value = np.nanmax(hist)
    params['P0'].min = 0.5*params['P0'].value


    fit_offset = 0.1*np.exp(-(a**2)/(4*np.log(2)))*np.nanmax(hist)+1
    result = mod_gauss.fit(hist[hist>=fit_offset],params,t=bins[hist>=fit_offset])
    
    P0_fit = result.params['P0']
    t0_fit = result.params['t0']
    t_FWHM_fit = result.params['t_FWHM']

    f_fit = lambda t: f_gauss(t,P0_fit,t0_fit,t_FWHM_fit)
    hist_fit = np.array([f_fit(t) for t in bins])
    
    if ploton==True:
        plt.figure()
        plt.plot(bins,hist)
        plt.plot(bins[hist>=fit_offset],hist[hist>=fit_offset])
        plt.plot(bins,hist_fit)
    
    # The thresholded image
    thresh_im = np.where((im>t0_fit+a*t_FWHM_fit) | (im<t0_fit-a*t_FWHM_fit),255,0)
    #thresh_im1 = np.where((im>t0_fit+a*t_FWHM_fit),255,0)
    #thresh_im = np.where((im<t0_fit-a*t_FWHM_fit),127,thresh_im1)

    thresh_im = np.array(thresh_im,dtype=np.uint8)

    return thresh_im,hist,bins,result



