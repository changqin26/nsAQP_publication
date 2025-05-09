"""
Created on Mar 21, 2015

Physical operator that implements a JOIN.
This implementation is an extension of the Xgjoin operator
(see ANAPSID https://github.com/anapsid/anapsid).

The intermediate results are stored in queues and processed incrementally.

@author: Maribel Acosta
"""

from multiprocessing import Value
from Queue import Empty
from operatorstructures import Tuple, Record, RJTTail
from time import time
from random import randint


class Fjoin(object):

    def __init__(self, id_operator, variables, eddies):
        self.left_table = dict()
        self.right_table = dict()
        self.id_operator = id_operator
        self.vars = set(variables)
        self.eof = Tuple("EOF", 0, 0, set(), self.id_operator)
        self.eddies = eddies
        self.eddy = randint(1, self.eddies)
        self.left = None
        self.right = None
        self.qresults = None
        self.probing = Value('i', 1)
        self.independent_inputs = 2
        self.probing_left = 1
        self.probing_right = 1

    def execute(self, inputs, out):

        # Initialize input and output queues.
        self.left = inputs[0]
        self.right = inputs[1]
        self.qresults = out

        # Get the tuples from the input queues.
        while True:

            self.probing.value = 1
            # Try to get and process tuple from left queue.
            try:
                self.probing_left = 1
                tuple1 = self.left.get(False)
                #"""
                if False: #("Cocaine" in str(tuple1.data)):
                    print("Left Queue")
                    print(tuple1.data)
                    print(tuple1.ready)
                    print(tuple1.done)
                    print(self.id_operator)
                    print()
                #"""
                self.stage1(tuple1, self.left_table, self.right_table)

            except Empty:
                # Empty: in tuple1 = self.left.get(False), when the queue is empty.
                self.probing_left = 0
                self.probing.value = self.probing_left | self.probing_right
                pass
            except TypeError:
                # TypeError: in resource = resource + tuple[var], when the tuple is "EOF".
                pass
            except IOError:
                # IOError: when a tuple is received, but the alarm is fired.
                pass

            # Try to get and process tuple from right queue.
            try:
                self.probing_right = 1
                tuple2 = self.right.get(False)
                """
                if ("Cocaine" in str(tuple2.data)):
                    print("Right Queue")
                    print(tuple2.data)
                    print(tuple2.ready)
                    print(tuple2.done)
                    print(self.id_operator)
                    print()
                """
                self.stage1(tuple2, self.right_table, self.left_table)
                #self.probing.value = 0

            except Empty:
                # Empty: in tuple2 = self.right.get(False), when the queue is empty.
                self.probing_right = 0
                self.probing.value = self.probing_left | self.probing_right
                pass
            except TypeError:
                # TypeError: in resource = resource + tuple[var], when the tuple is "EOF".
                pass
            except IOError:
                # IOError: when a tuple is received, but the alarm is fired.
                pass

    # Stage 1: While one of the sources is sending data.
    def stage1(self, tuple1, tuple_rjttable, other_rjttable):


        # Get the value(s) of the join variable(s) in the tuple.
        resource = ''
        if tuple1.data != "EOF":
            for var in self.vars:
                resource = resource + str(tuple1.data[var])
        else:
            resource = "EOF"

        # Probe the tuple against its RJT table.
        probe_ts = self.probe(tuple1, resource, tuple_rjttable)

        # Create the records.
        record = Record(tuple1, probe_ts, time(), float("inf"))

        # Insert the record in the corresponding RJT table.
        if resource in other_rjttable:
            other_rjttable.get(resource).updateRecords(record)
            other_rjttable.get(resource).setRJTProbeTS(probe_ts)
        else:
            tail = RJTTail(record, probe_ts)
            other_rjttable[resource] = tail

    # Stage 2: Executed when one source becomes blocked.
    def stage2(self, signum, frame):
        pass

    # Stage 3: Finalizes the join execution. It is fired when both sources has sent all the data.
    def stage3(self):
        return

    def probe(self, tuple1, resource, rjttable):

        # Probe a tuple against its corresponding table.
        probe_ts = time()

        # If the resource is in the table, produce results.
        if resource in rjttable:
            rjttable.get(resource).setRJTProbeTS(probe_ts)
            list_records = rjttable[resource].records

            #if ("Cocaine" in resource):
            #    print(list_records)

            # For each matching solution mapping, generate an answer.
            for record in list_records:
                if resource != "EOF":

                    # Check if tuples are compatible
                    compatible = True
                    for v in set(record.tuple.data.keys()) & set(tuple1.data.keys()):
                        if record.tuple.data[v] != tuple1.data[v]:
                            compatible = False
                            break

                    # Merge solution mappings, if compatible.
                    if compatible:
                        data = {}
                        data.update(record.tuple.data)
                        data.update(tuple1.data)

                        # Update ready and done vectors.
                        ready = record.tuple.ready | tuple1.ready
                        done = record.tuple.done | tuple1.done | pow(2, self.id_operator)
                        sources = list(set(record.tuple.sources) | set(tuple1.sources))

                        # Create tuple.
                        res = Tuple(data, ready, done, sources, self.id_operator)

                        # Send tuple to eddy operators.
                        self.qresults[self.eddy].put(res)
                else:
                    data = "EOF"
                    # Update ready and done vectors.
                    ready = record.tuple.ready | tuple1.ready
                    done = record.tuple.done | tuple1.done | pow(2, self.id_operator)
                    sources = list(set(record.tuple.sources) | set(tuple1.sources))

                    # Create tuple.
                    res = Tuple(data, ready, done, sources, self.id_operator)

                    # Send tuple to eddy operators.
                    self.qresults[self.eddy].put(res)

        return probe_ts
