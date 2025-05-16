#!/usr/bin/env python3
import rospy
import smach
from second_coursework.msg import FoodRequest

class WaitForRequest(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['received', 'abort'], output_keys=['name', 'food_item'])
        self.subscriber = rospy.Subscriber('/food_request', FoodRequest, self.request_callback)
        self.received_request = False
        self.request_name = None
        self.request_food = None

    def request_callback(self, msg):
        rospy.logwarn(f"Received request from {msg.name} for {msg.food_item}")
        self.received_request = True
        self.request_name = msg.name
        self.request_food = msg.food_item

    def execute(self, userdata):

        rospy.logwarn("Waiting for request...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.received_request:
                rospy.logwarn("Processing...")

                userdata.name = self.request_name
                userdata.food_item = self.request_food
                return 'received'
            rate.sleep()
        return 'abort'


