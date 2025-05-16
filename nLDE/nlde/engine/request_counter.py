"""
Created on Mar 15, 2023

"""

class RequestCounter:
    def __init__(self):
        self.counter = 0

    def increment(self):
        print("Incrementing request counter")
        self.counter += 1

    def get_count(self):
        return self.counter
