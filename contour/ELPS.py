import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from circle_fit import standardLSQ
from shapely.geometry import Polygon
rng = np.random.default_rng(seed=42) #...read up on this class

### Code made by Jonne Goedhart and Thijs Kuipers"

def fit_ellipse(x, y,use=''):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    if use!='RANSAC':
        return np.concatenate((ak, T @ ak)).ravel()
    else:
        return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def run_ellipse_rs(c,scale=1.0,ransac_opt=None,out_full=False): 
    if ransac_opt==None:
        ransac = False
    else:
        ransac = ransac_opt['on']
        try:
            if ransac_opt['sampsize_outlier'] is None:
                ransac_opt['sampsize_outlier'] = ransac_opt['sampsize']
        except KeyError:
            ransac_opt['sampsize_outlier'] = ransac_opt['sampsize']
        try:
            if ransac_opt['sampsize_inlier'] is None:
                ransac_opt['sampsize_inlier'] = ransac_opt['sampsize']
        except KeyError:
            ransac_opt['sampsize_inlier'] = ransac_opt['sampsize']


    # threshold in pixels #...adjust
 
    try:
        x = c[:,:,0].flatten()
        y = c[:,:,1].flatten()
    except TypeError:
        [x,y] = c
    except IndexError:
        x = c[:,0]
        y = c[:,1]

    data0 = np.array([x,y]).T
    data = np.copy(data0)

    val_list = []
    # Make first fit with all data
    x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1  = run_ellipse([data[:,0],data[:,1]],scale=scale)
    ret0 = x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1 
    if ransac==False:
        if out_full==False:
            return x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1
        elif out_full==True:
            return x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1, data0, None
    
    elif ransac==True:
        #try:
        data_fit0 = np.array([xfit1,yfit1]).T
        data_fit = np.copy(data_fit0)
        loss = ransac_opt['model_loss'](data,data_fit)
        res_sort = np.sort(np.copy(loss))
        i_end = int(np.floor(len(res_sort)*ransac_opt['fraction']))
        while i_end<20:
            ransac_opt['fraction'] = ransac_opt['fraction']**0.5
            i_end = int(np.floor(len(res_sort)*ransac_opt['fraction'])) #..adjust
        #lim = res_sort[i_end] # Cuts off at best point fraction

        thresholded = loss<ransac_opt['threshold']
        inliers = data0[thresholded]
        outliers = data0[~thresholded]
        val_list.append(np.nanmean(np.where(loss<ransac_opt['threshold'],loss,np.nan))/(np.sum(thresholded)**2))
        rng = np.random.default_rng(seed=0)
        for itern in range(0,ransac_opt['iterations']):
            # Add contour segment of length sampsize
            if len(outliers)>0:
                i_mid = np.random.randint(0,len(outliers))
                i_start = int(np.max([0,np.floor(i_mid-0.5*ransac_opt['sampsize_outlier'])]))
                i_end = int(np.min([len(outliers)-1,i_start+ransac_opt['sampsize_outlier']]))
                add_new = outliers[i_start:i_end]
            else:
                add_new = np.zeros((0,2))
            
            # Add old (inliers) contour segment
            if len(inliers)>0:
                i_mid = np.random.randint(0,len(inliers))
                i_start = int(np.max([0,np.floor(i_mid-0.5*ransac_opt['sampsize_inlier'])]))
                i_end = int(np.min([len(inliers)-1,i_start+ransac_opt['sampsize_inlier']]))
                add_old1 = inliers[:i_start]
                add_old2 = inliers[i_end:]
            elif len(outliers)>0:
                add_old1 = np.zeros((0,2))
                add_old2 = np.zeros((0,2))
            else:
                print('To do')
            
            maybe_inliers = np.concatenate((add_old1,add_old2,add_new))
            try:
                x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1  = run_ellipse([maybe_inliers[:,0],maybe_inliers[:,1]],scale=scale)
                data_fit = np.array([xfit1,yfit1]).T
                loss = ransac_opt['model_loss'](data0,data_fit)
                thresholded = loss<ransac_opt['threshold']
                val_new = np.nanmean(np.where(loss<ransac_opt['threshold'],loss,np.nan))/(np.sum(thresholded)**2)
                if val_new<val_list[-1]:
                    inliers = sort_elps_pnt(data0[thresholded],data_fit)
                    outliers = sort_elps_pnt(data0[~thresholded],data_fit)
                    #inliers = data0[thresholded]
                    #outliers = data0[~thresholded]
                    val_list.append(val_new)
                    print(val_list[-1],itern,end='\r')
                print('Performed ransac',end='\r')
            except Exception as e:
                print(str(e))

        if out_full==False:
            return x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1
        elif out_full==True:
            return x,y,xfit1,yfit1,[x01,y01,ap1,bp1,e1,phi1],res1, res_list1, inliers, outliers


        #except Exception as e:
        #    print(str(e))
        #    if out_full==False:
        #        return ret0
        #    elif out_full==True:
        #        return ret0, data0, None

def run_ellipse(c,scale=1.0):
    try:
        x = c[:,:,0].flatten()
        y = c[:,:,1].flatten()
    except TypeError:
        [x,y] = c
    coeffs = fit_ellipse(x*scale, y*scale)
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)

    x0 = x0/scale
    y0 = y0/scale
    ap = ap/scale
    bp = bp/scale

    xfit, yfit = get_ellipse_pts((x0, y0, ap, bp, e, phi))

    res = 0
    res_list = []
    for i in range(0,len(x)):
        xval = x[i]
        yval = y[i]
        
        resval = np.nanmin(((xfit-xval)**2+(yfit-yval)**2))
        res = res+resval
        res_list.append(res)
    res = ((res/len(x))**0.5)/((ap*bp)**0.5)

    return x,y,xfit,yfit,[x0,y0,ap,bp,e,phi],res,res_list

def run_circle(c):
    x = c[:,:,0]
    y = c[:,:,1]
    points = np.array([x,y]).T[0]
    x0, y0, ap, sigma = standardLSQ(points)
    bp = ap
    e = 1.0
    phi = 0.0
    xfit, yfit = get_ellipse_pts((x0, y0, ap, bp, e, phi))
    return x,y,xfit,yfit,[x0,y0,ap,bp,e,phi], np.nan,np.nan

def sort_elps_pnt(data,fit):
    i_sort = np.array([list(np.sum((val-fit)**2,axis=1)).index(np.nanmin(np.sum((val-fit)**2,axis=1))) for val in data])
    i_vec = np.array(range(0,len(i_sort)))
    order = [x for _,x in sorted(zip(i_sort,i_vec))]
    data_new = np.copy(data[order])

    return data_new

def check_cross(i,j,data_all,fit_all):
    #print(data_all[i])
    #print(fit_all[i])
    #print(data_all[j])
    #print(fit_all[j])

    p10 = Polygon(sort_elps_pnt(data_all[i],fit_all[i]))
    if fit_all[j] is None: #...should not happen but does!!!
        p20 = Polygon(sort_elps_pnt(data_all[j],fit_all[i]))
    else:
        p20 = Polygon(sort_elps_pnt(data_all[j],fit_all[i]))

    p1 = p10.buffer(0)
    p2 = p20.buffer(0)

    check = p1.intersects(p2)
    cross=False
    if check==True:
        area = p1.intersection(p2).area
        if area!=p1.area and area!=p2.area:
            cross=True

    return cross

class EllipsRegressor:
    def __init__(self, t):
        self.params_carthesian = None
        self.params_ellipse = None
        self.t = t
        from numpy.random import default_rng

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
        """
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        
        T = -np.linalg.solve(S3, S2.T)
#         print(T)
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
#         print(M, np.linalg.det(M))
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
#         print(con)
        ak = eigvec[:, np.nonzero(con > 0)[0]]
#         print(ak)
#         print(len(ak), ak.shape)
        if ak.shape[1] == 0 or ak.shape[1] == 2:
            return None
        
        self.params_carthesian = np.real(np.concatenate((ak, T @ ak)).ravel())
#         print(self.params_carthesian)
        try:
            self.__cart_to_pol()
        except ValueError:
            return None
        return self
    
    def __cart_to_pol(self):
        """

        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.

        """

        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = self.params_carthesian[0]
        b = self.params_carthesian[1] / 2
        c = self.params_carthesian[2]
        d = self.params_carthesian[3] / 2
        f = self.params_carthesian[4] / 2
        g = self.params_carthesian[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')

        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi

        self.params_ellipse = [x0, y0, ap, bp, e, phi] #output list --> differnt from function higher up in code

    #def fit(self, x: np.ndarray, y: np.ndarray):
    #    self.params_carthesian = fit_ellipse(x,y)
    #    try:
    #        self.params_ellipse = list(cart_to_pol(self.params_carthesian))
    #    except ValueError:
    #        return None
    #    return self


    def predict(self): #Get ellipse points (hetzelfde, generates ellipse)
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.

        """
        x0, y0, ap, bp, e, phi = self.params_ellipse
        # A grid of the parametric variable, t.

        x = x0 + ap * np.cos(self.t) * np.cos(phi) - bp * np.sin(self.t) * np.sin(phi)
        y = y0 + ap * np.cos(self.t) * np.sin(phi) + bp * np.sin(self.t) * np.cos(phi)
        return np.array([x, y]).T

    def loss(self, data):
        """
        normalized abs error
        # `loss`: function that returns a vector
        """
        ap, bp = self.params_ellipse[2:4] # Semi axis, wordt niet gebruikt hier
        loss = self.__error(data)
        return np.sqrt(loss) #/ np.sqrt((ap * bp)) #SEEMS EASIER TO TWEAK WITH THE THRESHOLD

    def metric(self, data, threshold):
        """
        normalized root mean square error
        # `metric`: function that returns a float
        """
        x0, y0, ap, bp = self.params_ellipse[:4]
        centroid_distance = np.sqrt(np.sum((data - np.array([x0,y0]))**2, 1)) #aanpassen naar N points
        metric = (np.sqrt(self.__error(data)) < threshold) / centroid_distance #Selects the points on the ellipse, division due to underestimating small ellipses
        return len(data) - sum(metric) #Aantal punten niet op ellipse

    def __error(self, data):
        ellips = self.predict()
        error = (data.reshape(-1,1,2) - ellips)**2
        error = np.sum(error, 2)
        error = np.min(error, 1) #Minimum disctance from point to ellipse
        return error

class RANSAC:
    def __init__(self, n=6, k=1000, k2=50, threshold=0.05, d=0.2, centroid_distance=10, number_of_clusters=5, max_elasticity=0.5, model=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.k2 = k2            # `k`: Maximum outer iterations allowed #Aantal 
        self.threshold = threshold # `t`: Threshold value to determine if points are fit well in number of pixels
        self.d = d              # `d`: minimum percentage of close data points required to assert model fits well (fraction of datapints on ellipse within threshold)
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.best_fits = [None for _ in range(number_of_clusters)] #per cluster best model fit fron ellipse model
        self.best_errors = [np.inf for _ in range(number_of_clusters)] #per cluster best error
        self.centroid_distance = centroid_distance #Maximum distance from ellipse center to input centroid --> scale to pix_size
        self.n_cluster = number_of_clusters #Number of clusters/ellipses --> Gan do smarter/dynamic
        self.max_elasticity = max_elasticity #max e value

    def fit(self, X, centroid, cont):
        '''
        X = data --> All edges
        Check Xy/Yx swap (according to Jonne)
        centroid: Expected centroid --> EA adjust to last image!!!
        cont: Contours found by:
        contrast = 1.0
out_sort1, q, edges1 = CONTOUR.get_all_contours(image,operator,th1=th1,th2=th2,center_off=center_off,step=step,fitrun=False,inside=True,inside_try=False,edge_val=edge_val,origin_val=origin_val,pixmax=pixmax,circularity_min=0.0,split_contours=False,point_min=10,contrast=contrast)
contrast = 3.0
out_sort2, q, edges2 = CONTOUR.get_all_contours(image,operator,th1=th1,th2=th2,center_off=center_off,step=step,fitrun=False,inside=True,inside_try=False,edge_val=edge_val,origin_val=origin_val,pixmax=pixmax,circularity_min=0.0,split_contours=False,point_min=10,contrast=contrast)

        --> So initialization of clusters
        '''
        self.clusters = np.zeros((len(X), self.n_cluster)) #Expeted 1 at every rom (one hot matrix)

        for iteration in range(self.k2):
            for cluster_id in range(self.n_cluster):
                self.clusters[:, cluster_id] = 0 #Reset
                self.undifined_points = X[~np.any(self.clusters, 1)] #Find points not yet in other cluster
                #print(self.undifined_points)
                self.best_errors[cluster_id] = np.inf #reset the cluster error

                for _ in range(self.k):
                    if iteration > 0:
                        maybe_inliers = rng.choice(self.undifined_points, size=self.n, replace=False)
                    else:
                        #Initialization first iteration
                        maybe_inliers = rng.choice(cont[cluster_id], size=self.n, replace=False) # THIS SHOULD USE YOUR CONTOURS BUT IT KEEPS BREAKING THE ELLIPSE FUNCTION Important of cont order (Ik denk dat ze georteerd zijn op area (grootste eerst)
                        maybe_inliers = np.roll(maybe_inliers, 1, axis=1) #Switches X and Y axis --> Maybe wrong with edges (EA change)

                    maybe_model = copy(self.model).fit(maybe_inliers[:,0], maybe_inliers[:,1])
                    if maybe_model is None:
                        continue #Go to next instance of loop
                    
                    thresholded = maybe_model.loss(self.undifined_points) < self.threshold #Area of boolen
                    model_centroid = maybe_model.params_ellipse[:2]
                    elasticity = maybe_model.params_ellipse[-2] #Maybe also use area
                    #Area toeveoegen

                    if (np.sum(thresholded) / len(self.undifined_points) > self.d and 
                            np.sqrt(np.sum((centroid - model_centroid)**2)) < self.centroid_distance and 
                            elasticity < self.max_elasticity): #Ignore fits with center_offset to big and number of points too little

                        inliers = self.undifined_points[thresholded]
                        better_model = copy(self.model).fit(inliers[:,0], inliers[:,1]) #Make op basis van all inliers

                        this_error = better_model.metric(self.undifined_points, self.threshold)
                        if this_error < self.best_errors[cluster_id]:
                            self.best_errors[cluster_id] = this_error
                            self.best_fits[cluster_id] = better_model

                if self.best_fits[cluster_id] is not None:
                    best_threshold = np.logical_and(self.best_fits[cluster_id].loss(X) < self.threshold, ~np.any(self.clusters, 1)) #Selects all points within threshold and not assigned to different cluseter --> inliers
                    self.clusters[best_threshold, cluster_id] = 1 #one hot matrix

            #predictions = self.predict()
            #for i,p in enumerate(predictions):
            #    #plt.imshow(image, cmap='gray')
            #    #plt.imshow(edges_image_new, cmap='gray', alpha=0.5)
            #    plt.plot(centroid, centroid, '*',color='y')
            #    plt.plot(p[:,1], p[:,0], c='C'+str(i), linewidth=0.5)
            #    points = self.get_cluster_points(X, i)
            #    plt.scatter(points[:,1], points[:,0], c='C'+str(i), s=2)
            #plt.show()

            #Convergence theory space
            #Convergence check
            #Splitten of mergen van clusers
            #break statement
            # Test if intersecting --> Else delete

        self.undifined_points = X[~np.any(self.clusters, 1)] #Points not on edges (contours)
        return self

    def get_cluster_points(self, X, cluster):
        try:
            self.edgepoints
        except:
            edgepoints = self.get_all_cluster_points(X)

        return self.edgepoints[cluster]

    def predict(self):
        ''' Generates al found ellips models'''
        return [best_fit.predict() for best_fit in self.best_fits if best_fit is not None]

    def get_all_cluster_points(self,X):
        ret = []
        for i in range(0,len(self.best_fits)):
            if self.best_fits[i] is not None:
                val = X[self.clusters[:, i] >= 1]
                ret.append(val)

        self.edgepoints = ret
        
        return ret
