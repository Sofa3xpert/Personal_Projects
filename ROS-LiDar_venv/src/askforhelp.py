#!/usr/bin/env python3
import rospy
import sys
import smach
from std_msgs.msg import String


class AskForHelp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['help_received', 'help_failed'])
        self.speech_sub = rospy.Subscriber('/speech_recognition/final_result', String, self.speech_rec)
        self.tts_pub = rospy.Publisher('/tts/phrase', String, queue_size=1)
        self.help_status = False

    def speech_rec(self, msg): # modify in case of complex behavior
        # assume we always get 'help' and status is always true:
        self.help_status = True

    def execute(self, ud):
        self.tts_pub.publish("Can you please help me")
        if self.help_status:
            self.tts_pub.publish("Going back to the table")
            rospy.logwarn("Returning Online")
            return 'help_received'
        return 'help_failed'


