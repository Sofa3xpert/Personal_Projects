#!/usr/bin/env python3
import rospy
import smach
import math
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class ReturnToTable(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['arrived', 'failed'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def execute(self, ud):
        rospy.logwarn("Returning to Table...")
        return self.send_goal_to_table()

    def send_goal_to_table(self):
        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = 6.0
        goal.target_pose.pose.position.y = 2.0


        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(-math.pi / 4)
        goal.target_pose.pose.orientation.w = math.sin(-math.pi / 4)

        rospy.logwarn("Sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()

        if self.client.get_state() == 3:
            rospy.logwarn("Successfully Returned to table")
            return 'arrived'
        else:
            rospy.logwarn("Failed to return to table")
            return 'failed'
