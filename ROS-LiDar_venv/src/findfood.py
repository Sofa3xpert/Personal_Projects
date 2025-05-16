#!/usr/bin/env python3
import rospy
import smach
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from yolov4 import Detector
from second_coursework.msg import FoodDetection
from std_msgs.msg import String


class FindFood(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['found', 'not_found'], input_keys=['food_item'])
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.tts_pub = rospy.Publisher('/tts/phrase', String, queue_size=1)
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.bridge = CvBridge()
        self.detector = Detector(gpu_id=0, config_path='/opt/darknet/cfg/yolov4.cfg',
                                 weights_path='/opt/darknet/yolov4.weights',
                                 lib_darknet_path='/opt/darknet/libdarknet.so',
                                 meta_path='/home/k22038605/rosWS/src/second_coursework/coco.data') #please change in case of transfer/marking
        self.food_found = None
        self.target_food = None
        self.lastFrame = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if cv_image is not None:
            cv_copy = cv_image.copy()
            img_arr = cv2.resize(cv_image, (self.detector.network_width(), self.detector.network_height()))
            detections = self.detector.perform_detect(image_path_or_buf=img_arr, show_image=True)

            for detection in detections:
                if detection.class_name == 'diningtable':
                    detection.class_name = 'pizza'
                rospy.loginfo(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % |')
                d = FoodDetection(detection.class_name, detection.class_confidence, detection.left_x, detection.top_y,
                                  detection.width, detection.height)
                # if d.name in ['pizza', 'sandwich', 'broccoli']: # testing
                rospy.logwarn(f"Detected {detection.class_name}")
                if d.name == self.target_food:
                    self.food_found = True

    def execute(self, userdata):

        self.target_food = userdata.food_item
        rospy.logwarn(f"Searching for {self.target_food}...")
        rate = rospy.Rate(10)
        time_limit = rospy.Time.now() + rospy.Duration(30)

        while not rospy.is_shutdown() and rospy.Time.now() < time_limit:
            if self.food_found:
                rospy.logwarn(f"{self.target_food} found!")
                self.stop_spinning()
                return 'found'
            self.spin()
            rate.sleep()

        self.stop_spinning()
        rospy.logwarn(f"{self.target_food} not found")
        message = f"I am sorry. Failed to deliver food: {self.target_food}"
        self.tts_pub.publish(message)
        return 'not_found'

    def spin(self):
        twist = Twist()
        twist.angular.z = 0.5
        self.vel_pub.publish(twist)

    def stop_spinning(self):
        self.vel_pub.publish(Twist())
