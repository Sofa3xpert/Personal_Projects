#!/usr/bin/env python3
import rospy
import smach
import math
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class GoToTable(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failed'])
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def execute(self, userdata):
        rospy.logwarn("Going to Table...")
        return self.send_goal_to_table()

    def send_goal_to_table(self):
        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        #coords of 'table'
        goal.target_pose.pose.position.x = 6.0
        goal.target_pose.pose.position.y = 2.0
        goal.target_pose.pose.position.z = 0

        goal.target_pose.pose.orientation.x = 0
        goal.target_pose.pose.orientation.y = 0
        goal.target_pose.pose.orientation.z = math.sin(-math.pi / 4)
        goal.target_pose.pose.orientation.w = math.sin(-math.pi / 4)


        rospy.logwarn("Sending goal")
        self.client.send_goal(goal)

        wait = self.client.wait_for_result()

        if not wait:
            rospy.logwarn("Failed to Reach table")
            return 'failed'
        else:
            result = self.client.get_state()
            if result == 3:
                rospy.logwarn("Successfully Reached table")
                return 'success'
            else:
                rospy.logwarn("Failed to Reach table")
                return 'failed'
