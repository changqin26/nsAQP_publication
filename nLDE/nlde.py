#!/usr/bin/env python

"""
Created on Mar 25, 2015

@author: Maribel Acosta
"""
import errno

from nlde.engine.eddynetwork import EddyNetwork
from nlde.policy.nopolicy import NoPolicy
from nlde.policy.ticketpolicy import TicketPolicy
from nlde.policy.uniformrandompolicy import UniformRandomPolicy
from nlde.policy.productivepolicy import ProductivePolicy
from nlde.policy.includingNLJpolicy import IncludingNLJ
from nlde.policy.productivepolicyNLJ import ProductivePolicyNLJ
from nlde.policy.uniformrandompolicyNLJ import UniformRandomPolicyNLJ
from nlde.policy.ticketpolicyNLJ import TicketPolicyNLJ
from nlde.policy.heavyuniformrandompolicy import HeavyRandomPolicy
from nlde.policy.heavyuniformrandompolicyNLJ import HeavyRandomPolicyNLJ
# from nlde.engine.contactsource import NLDERequestCounter

import argparse
import os
import signal
import sys
import traceback  # Newly added
from multiprocessing import active_children, Queue
from time import time
import logging  # Newly added
import datetime
import threading

def list_active_threads():
    active_threads = threading.enumerate()
    logging.info("Active Threads: {0}".format(active_threads))

def get_options():
    parser = argparse.ArgumentParser(description="nLDE: An engine to execute "
                                                 "SPARQL queries over Triple Pattern Fragments")

    # nLDE arguments.
    parser.add_argument("-s", "--server",
                        help="URL of the triple pattern fragment server (required)")
    parser.add_argument("-f", "--file",
                        help="file name of the SPARQL query (required, or -q)")
    parser.add_argument("-q", "--query",
                        help="SPARQL query (required, or -f)")
    parser.add_argument("-r", "--results",
                        help="format of the output results",
                        choices=["y", "n", "all"],
                        default="y")
    parser.add_argument("-e", "--eddies",
                        help="number of eddy processes to create",
                        type=int,
                        default=2)
    parser.add_argument("-p", "--policy",
                        help="routing policy used by eddy operators",
                        choices=["NoPolicy", "Ticket", "Random", "Productivity", "MiniOutputFirst", "ProNLJ", "RanNLJ",
                                 "TpNLJ", "HeavyRandom", "HeavyRanNLJ"],
                        default="NoPolicy")
    parser.add_argument("-t", "--timeout",
                        help="query execution timeout",
                        type=int)
    parser.add_argument("-x", "--explain")

    """
    parser.add_argument("-c", "--print-request-count", action="store_true",
                        help="print the total number of requests made during query execution")
    """
    args = parser.parse_args()

    # Handling mandatory arguments.
    err = False
    msg = []
    if not args.server:
        err = True
        msg.append("error: no server specified. Use argument -s to specify the address of a server.")

    if not args.file and not args.query:
        err = True
        msg.append("error: no query specified. Use argument -f or -q to specify a query.")

    if err:
        parser.print_usage()
        print
        "\n".join(msg)
        sys.exit(1)

    return args.server, args.file, args.query, args.eddies, args.timeout, args.results, args.policy, args.explain


class NLDE(object):
    def __init__(self, source, queryfile, query, eddies, timeout, printres, policy_str, explain):

        self.source = source
        self.queryfile = queryfile
        self.query = query
        self.query_id = ""
        self.eddies = eddies
        self.timeout = timeout
        self.printres = printres
        self.policy_str = policy_str
        self.explain = explain
        self.network = None
        self.p_list = None
        self.res = Queue()
        self.timeout_occurred = False
        self.subprocesses = []

        # Open query.
        if self.queryfile:
            self.query = open(self.queryfile).read()
            self.query_id = self.queryfile[self.queryfile.rfind("/") + 1:]

        # Set routing policy.
        if self.policy_str == "NoPolicy":
            self.policy = NoPolicy()
        elif self.policy_str == "Ticket":
            self.policy = TicketPolicy()
        elif self.policy_str == "Random":
            self.policy = UniformRandomPolicy()
        elif self.policy_str == "Productivity":
            self.policy = ProductivePolicy()
        elif self.policy_str == "MiniOutputFirst":
            self.policy = IncludingNLJ()
        elif self.policy_str == "TpNLJ":
            self.policy = TicketPolicyNLJ()
        elif self.policy_str == "RanNLJ":
            self.policy = UniformRandomPolicyNLJ()
        elif self.policy_str == "ProNLJ":
            self.policy = ProductivePolicyNLJ()
        elif self.policy_str == "HeavyRandom":
            self.policy = HeavyRandomPolicy()
        elif self.policy_str == "HeavyRanNLJ":
            self.policy = HeavyRandomPolicyNLJ()

        # Set execution variables.
        self.init_time = None
        self.time_first = None
        self.time_total = None
        self.card = 0
        self.xerror = ""
        self.requests = 0
        self.intermediate_results = 0

        # Set execution timeout.
        if self.timeout:
            signal.signal(signal.SIGALRM, self.call_timeout)
            signal.alarm(self.timeout)
            # Added more logging to track the flow
            #logging.info("NLDE object initialized")

    def execute(self):
        try:
            self.init_time = time()

        except Exception as e:
            traceback.print_exc()

        try:
            # Create eddy network.
            network = EddyNetwork(self.query, self.policy, source=self.source, n_eddy=self.eddies, explain=self.explain)
            self.network = network
            self.p_list = network.p_list

            if self.printres == "y":
                self.print_solutions(network)
            elif self.printres == "all":
                self.print_all(network)
            else:
                self.print_basics(network)

            self.requests = network.request_counter.value

            try:
                self.intermediate_results = network.intermediate_results_counter.value
            except Exception as e:
                traceback.print_exc()

            self.summary()

        except Exception as e:
            traceback.print_exc()  # Print stack trace for detailed error info


    # Print only basic stats, but still iterate over results.
    def print_basics(self, network):

        network.execute(self.res)

        # Handle the first query answer.
        ri = self.res.get(True)
        self.time_first = time() - self.init_time
        count = 0
        if ri.data == "EOF":
            count = count + 1
        else:
            self.card = self.card + 1

        # Handle the rest of the query answer.
        while count < network.n_eddy:
            ri = self.res.get(True)
            if ri.data == "EOF":
                count = count + 1
            else:
                self.card = self.card + 1

        self.time_total = time() - self.init_time

    # Print only solution mappings.
    def print_solutions(self, network):

        network.execute(self.res)

        # Handle the first query answer.
        ri = self.res.get(True)
        self.time_first = time() - self.init_time
        count = 0
        if ri.data == "EOF":
            count = count + 1
        else:
            self.card = self.card + 1
            print
            str(ri.data)

        # Handle the rest of the query answer.
        while count < network.n_eddy:
            ri = self.res.get(True)
            if ri.data == "EOF":
                count = count + 1
            else:
                self.card = self.card + 1
                print
                str(ri.data)

        self.time_total = time() - self.init_time

    # Print all stats for each solution mapping.
    def print_all(self, network):

        network.execute(self.res)

        # Handle the first query answer.
        ri = self.res.get(True)
        self.time_first = time() - self.init_time
        count = 0
        if ri.data == "EOF":
            count = count + 1
        else:
            self.card = self.card + 1
            print
            self.query_id + "\t" + str(ri.data) + "\t" + str(self.time_first) + "\t" + str(self.card) + "\t" + str(ri)

        # Handle the rest of the query answer.
        while count < network.n_eddy:
            ri = self.res.get(True)
            if ri.data == "EOF":
                count = count + 1
            else:
                self.card = self.card + 1
                t = time() - self.init_time
                print
                self.query_id + "\t" + str(ri.data) + "\t" + str(t) + "\t" + str(self.card) + "\t" + str(ri)

        self.time_total = time() - self.init_time

    # Final stats of execution.
    def summary(self):

        print(self.query_id + "\t" + str(self.time_first) + "\t" + str(self.time_total) +
              "\t" + str(self.card) + "\t" + str(self.xerror) + "\t" + str(self.requests) +
              "\t" + str(self.intermediate_results))
        """
        print self.query_id + "\t" + str(self.time_first) + "\t" + str(self.time_total) + \
        "\t" + str(self.card) + "\t" + str(self.xerror) + "\t" + str(self.requests)  # Intermediate results removed
        """


    # Timeout was fired.
    def call_timeout(self, sig, err):
        try:
            #logging.warning("Timeout occurred after {} seconds".format(self.timeout))
            timeout_start = time()  # Start time for entire timeout handling

            # Marking that timeout has occurred
            self.timeout_occurred = True

            list_active_threads()
            # Update time_total
            self.time_total = time() - self.init_time

            self.finalize()

            # Update intermediate_results

            self.intermediate_results = self.network.intermediate_results_counter.value
            # Update requests
            self.requests = self.network.request_counter.value

            # Summary method

            self.summary()

        except Exception as e:
            #logging.error("Exception in call_timeout: {}".format(e))
            traceback.print_exc()  # Log the detailed exception info




        #tells when the Python script finishes its execution.
        sys.exit(1)  # Exiting due to timeout

    # Finalize execution: kill sub-processes.

    def finalize(self):
        self.res.close()

        list_active_threads()

        while not self.p_list.empty():
            try:
                pid = self.p_list.get(timeout=1)  # Adding a timeout for safety
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as e:
                    if e.errno != errno.ESRCH:  # Ignore 'No such process' error
                        pass
            except EOFError:
                break  # Exit the loop if the queue is empty or closed

        for p in active_children():
            try:
                p.terminate()
                p.join()
            except OSError as e:
                if e.errno != errno.ESRCH:
                    pass



if __name__ == '__main__':
    startTime = datetime.datetime.now()
    try:


        (fragment, queryfile, query, eddies, timeout, printres, policy_str, explain) = get_options()

        #logging.info("Starting NLDE with fragment: {}, queryfile: {}, timeout: {}".format(fragment, queryfile, timeout))

        init_start = time()
        nlde = NLDE(fragment, queryfile, query, eddies, timeout, printres, policy_str, explain)
        init_end = time()

        execute_start = time()
        nlde.execute()
        execute_end = time()

        finalize_start = time()
        nlde.finalize()
        finalize_end = time()

        #logging.info("NLDE execution completed")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)  # Exiting due to an exception


    finally:

        endTime = datetime.datetime.now()
    # print("*******")

    sys.exit(0)  # Normal exit
