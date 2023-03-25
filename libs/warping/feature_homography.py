import numpy as np
import cv2
import imutils


class HandCraftedFeatureHomography:
    def __init__(self,
                 short_edge_size=600,
                 min_match_count=10,
                 flann_index_kdtree=1,
                 descriptor_method='sift',
                 lowe_threshold=0.8):

        if descriptor_method == 'sift':
            self.descriptor = cv2.SIFT_create()
        elif descriptor_method == 'orb':
            self.descriptor = cv2.ORB_create()
        else:
            raise ValueError

        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_match_count = min_match_count

        self.short_edge_size = short_edge_size
        self.lowe_threshold = lowe_threshold

    def detect_kp_and_compute_features(self, img):
        kp, desc = self.descriptor.detectAndCompute(img, None)
        return kp, desc

    def resize_image(self, img):
        h, w = img.shape[:2]
        if h > w:
            img = imutils.resize(img, width=self.short_edge_size)
            ratio = w / self.short_edge_size
        else:
            img = imutils.resize(img, height=self.short_edge_size)
            ratio = h / self.short_edge_size
        return img, ratio

    def find_homography(self, ref_img, query_img, draw_match_save=''):
        ref_img, ratio_ref = self.resize_image(ref_img)
        query_img, ratio_query = self.resize_image(query_img)

        ref_kps, ref_desc = self.detect_kp_and_compute_features(ref_img)
        query_kps, query_desc = self.detect_kp_and_compute_features(query_img)

        matches = self.flann_matcher.knnMatch(ref_desc, query_desc, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < self.lowe_threshold * n.distance:
                good.append(m)

        if len(good) > self.min_match_count:
            ref_pts = np.float32([ref_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            query_pts = np.float32([query_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # H, mask = cv2.findHomography(ref_pts, query_pts, cv2.RANSAC, 5.0)
            H, mask = cv2.findHomography(query_pts, ref_pts, cv2.RANSAC, 5.0)

            if draw_match_save:
                matchesMask = mask.ravel().tolist()
                h, w = ref_img.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                query_img = cv2.polylines(query_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                img3 = cv2.drawMatches(ref_img, ref_kps, query_img, query_kps, good, None, **draw_params)
                cv2.imwrite(draw_match_save, img3)

            h, w = ref_img.shape[:2]
            warped_query = cv2.warpPerspective(query_img, H, (w, h))
            return H, warped_query, (ratio_ref, ratio_query)

        print("Not enough matches are found - {}/{}".format(len(good), self.min_match_count))
        return None


"""
if __name__ == '__main__':
    sift_homography = SIFTHomography(descriptor_method='sift')
    query_img = cv2.imread('/media/LinixData1/cucumber/mini_dataset/fruit_norm/20220407_205533.jpg')  # queryImage
    cv2.imwrite('query_img.png', query_img)
    # query_img = cv2.imread('/media/LinixData1/cucumber/mini_dataset/leaf_ht/20220426_011752.jpg')  # queryImage
    ref_img = cv2.imread('data/board_template_jpg/A1-2022-09-15-0002.jpg')  # trainImage
    H, warped_query, _ = sift_homography.find_homography(ref_img.copy(), query_img.copy(), 'matching.jpg')
    cv2.imwrite('warped_query.png', warped_query)
"""

