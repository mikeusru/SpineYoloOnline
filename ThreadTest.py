from threading import Thread
from random import randint
import time


class MyThread(Thread):

    def __init__(self, val, pusher):
        ''' Constructor. '''
        Thread.__init__(self)
        self.val = val
        self.pusher = pusher

    def run(self):
        for i in range(1, self.val):
            # print('Value %d in thread %s' % (i, self.getName()))
            self.pusher.trigger(u'message', u'send', {
                u'name' : 'thread poster',
                u'message' : 'Value %d in thread %s' % (i, self.getName())
            })
            # Sleep for random time between 1 ~ 3 second
            secondsToSleep = randint(1, 5)
            self.pusher.trigger(u'message', u'send', {
                u'name': 'thread poster',
                u'message': 'Value %d in thread %s' % (i, self.getName())
            })
            time.sleep(secondsToSleep)