#!/usr/bin/env python3
import rospy
import sys
import smach
from std_msgs.msg import String
from second_coursework.msg import FoodRequest

class DeliverFood(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['delivered', 'failed'], input_keys=['name','food_item'])
        self.tts_pub = rospy.Publisher('/tts/phrase', String, queue_size=1)
        self.food_sub = rospy.Subscriber('/food_detection', FoodRequest, queue_size=10)


    def execute(self, ud):
        rospy.logwarn("Delivering Food...")
        name = ud.name
        food = ud.food_item
        try:
            message = f"Hello {name}, here is your food: {food}"
            self.tts_pub.publish(message)
            rospy.sleep(2)
            return 'delivered'
        except Exception as e:
            message = f"I am sorry {name}. Failed to deliver food: {str(e)}"
            self.tts_pub.publish(message)
            rospy.sleep(2)
            return 'failed'
