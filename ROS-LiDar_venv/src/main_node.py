#!/usr/bin/env python3
import rospy
import sys
import smach
import smach_ros

from gotokitchen import GoToKitchen
from waitforrequest import WaitForRequest
from gototable import GoToTable
from findfood import FindFood
from returntotable import ReturnToTable
from deliverfood import DeliverFood
from askforhelp import AskForHelp

def main():


    sm = smach.StateMachine(outcomes=['task_completed'])

    with sm:

        sm.userdata.sm_food_item = None
        sm.userdata.sm_name = None

        smach.StateMachine.add('GOTOTABLE', GoToTable(),
                               transitions={'success': 'WAITFORREQUEST',
                                            'failed': 'task_completed'})
        smach.StateMachine.add('WAITFORREQUEST', WaitForRequest(),
                               transitions={'received': 'GOTOKITCHEN',
                                            'abort': 'task_completed'},
                               remapping={'food_item': 'sm_food_item',
                                          'name': 'sm_name'})
        smach.StateMachine.add('GOTOKITCHEN', GoToKitchen(),
                               transitions={'arrived': 'FINDFOOD',
                                            'failed': 'task_completed'})
        smach.StateMachine.add('FINDFOOD', FindFood(),
                               transitions={'found': 'ASKFORHELP',
                                            'not_found': 'GOTOTABLE'},
                               remapping={'food_item': 'sm_food_item'})
        smach.StateMachine.add('ASKFORHELP', AskForHelp(),
                               transitions={'help_received': 'RETURN_TO_TABLE',
                                            'help_failed': 'task_completed'})
        smach.StateMachine.add('RETURN_TO_TABLE', ReturnToTable(),
                               transitions={'arrived': 'DELIVERFOOD',
                                            'failed': 'task_completed'})
        smach.StateMachine.add('DELIVERFOOD', DeliverFood(),
                               transitions={'delivered': 'WAITFORREQUEST',
                                            'failed': 'WAITFORREQUEST'},
                               remapping={'food_item': 'sm_food_item',
                                          'name': 'sm_name'})



    outcome = sm.execute()

if __name__ == '__main__':
    rospy.init_node('robot_state_machine')
    while not rospy.is_shutdown():
        main()
        rospy.spin()
