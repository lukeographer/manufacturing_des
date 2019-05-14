import simpy
import statistics
import matplotlib.pyplot as plt
import numpy.random as random
import numpy
import math

# simulation environment

#ToDO:
#Add comparison between two to find payback time

entity_name = ''
server_name = ''
server_capacity = ''
entity_number = ''
num_runs = ''

# create model inputs
model_inputs = {}
simChoice = int(input("Press 1 for technician. \nPress 2 for Cobot.\n"))
if(simChoice == 1):
    with open("tech_inputs.txt") as input_file:
        for line in input_file:
            (key, val) = line.split()
            model_inputs[key] = val
            if "attribute" in key: print(key, val)
    flag = True
elif(simChoice == 2):
    with open("cobot_inputs.txt") as input_file:
        for line in input_file:
            (key, val) = line.split()
            model_inputs[key] = val
            if "attribute" in key: print(key, val)
    flag = True
else:
    print("Please enter valid input.")
    exit(-1)
print(model_inputs)

# Create variables from dictionary key names and assign their values, e.g. server_name=gas_station.
for key, value in model_inputs.items():
    exec(key + '=value')

print(f'The {operator_name} serves entity {entity_name} at {server_name1} and then {server_name2}')

# read random times and populate lists

entity_number = int(entity_number)
num_runs = int(num_runs)

# simulation statistics
simMeanWaitingTimes = []
simVarianceWaitingTimes = []
simIdleTimes = []
simTotalQueueTime = []
simTotalTime = []
simMinQueue = []
simMaxQueue = []
simMeanQueue = []
simResourceUtil = []
simResourceComp = []
simRejectRatio = []
simCost = []
simCostPerEntity = []

# statistics arrays
interarrivalTimes = []
serviceTimesLayup = []
serviceTimesCompact = []
rejectPartsLayup = []
rejectPartsCompact = []
waitingTimesLayup = []
waitingTimesCompact = []
queueLengthsLayup = []
queueLengthsCompact = []
idleTimesLayup = []
idleTimesCompact = []
exitTimesLayup = []
exitTimesCompact = []
arrivingTimes = []



# monitor queue
def monitorLayupQueue(resource):
    """This monitors the resource level for queueLength"""

    # print('Queue size: %s' % len(resource.queue))
    queueLengthsLayup.append(len(resource.queue))

    # monitor queue
def monitorCompactQueue(resource):
    """This monitors the resource level for queueLength"""

    # print('Queue size: %s' % len(resource.queue))
    queueLengthsCompact.append(len(resource.queue))

    # server process Generator
def currProcess(environment, name, serverStation1, serverStation2, arrivalTime,
                serviceTimesLayup, serviceTimesCompact, rejectedPartLayup,
                rejectedPartCompact):
    # event triggered at server
    yield environment.timeout(arrivalTime)
    arrivingTimes.append(environment.now)
    # request Layup resource allocation
    with serverStation1.request() as request:
        yield request
        # calculate waiting time
        waitingTimeLayup = environment.now - arrivalTime
        # serve entity
        yield environment.timeout(serviceTimesLayup)
        exitTimesLayup.append(environment.now)
        currExit = environment.now;
        # record waiting time
        waitingTimesLayup.append(waitingTimeLayup)
        # monitor Queue Length
        monitorLayupQueue(serverStation1)
        if(rejectedPartLayup == True):
           serviceTimesCompact = 0
    # request Compact resource allocation
    with serverStation2.request() as request:
        yield request
        # calculate waiting time
        waitingTimeCompact = environment.now - currExit
        # serve entity
        yield environment.timeout(serviceTimesCompact)
        exitTimesCompact.append(environment.now)
        # record waiting time
        waitingTimesCompact.append(waitingTimeCompact)
        # monitor Queue Length
        monitorCompactQueue(serverStation1)
        #if (rejectedPartCompact == True):
         #   print('%s was rejected after Compact Process' % (name))


for p in range(num_runs):
    environment = simpy.Environment()
    # print(" ")
    for x in range(entity_number):
        serviceTimesLayup.append(random.triangular(float(layup_min), float(layup_mode), float(layup_max)))
        if (simChoice == 2):
            serviceTimesCompact.append(random.uniform(float(compact_min), float(compact_max)))
        else:
            serviceTimesCompact.append(random.triangular(float(compact_min), float(compact_mode), float(compact_max)))
        rejectPartsLayup.append(random.triangular(float(reject_min_layup), float(reject_mode_layup),
                                                  float(reject_max_layup)) > random.uniform(0,1))
        rejectPartsCompact.append(random.triangular(float(reject_min_compact), float(reject_mode_compact),
                                                    float(reject_max_compact)) > random.uniform(0, 1))
    # copies layup service times to interarrival times to there is no queue for the layup
    interarrivalTimes = serviceTimesLayup.copy();
    interarrivalTimes.insert(0,0) #start arriving at time 0
    del(interarrivalTimes[-1]) #remove the last index

    serviceTimesLayup = [float(i) for i in serviceTimesLayup]
    serviceTimesCompact = [float(i) for i in serviceTimesCompact]

    server1Capacity = int(server1_capacity)
    server2Capacity = int(server2_capacity)
    # server station
    serverStation1 = simpy.Resource(environment, capacity=server1Capacity)
    serverStation2 = simpy.Resource(environment, capacity=server2Capacity)

    arrivalTime = 0

    # simulate server/queue system
    for i in range(len(interarrivalTimes)):
        arrivalTime += interarrivalTimes[i]
        serveTimeLayup = serviceTimesLayup[i]
        serveTimeCompact = serviceTimesCompact[i]
        rejectedPartLayup = rejectPartsLayup[i]
        rejectedPartCompact = rejectPartsCompact[i]

        environment.process(currProcess(environment, 'Entity %d' % i, serverStation1, serverStation2,
                                        arrivalTime, serveTimeLayup, serveTimeCompact, rejectedPartLayup,
                                        rejectedPartCompact))
    # run simulation
    environment.run()

    # calculate idle time
    for i in range(len(arrivingTimes) - 1):
        idleTimeLayup = max(0, arrivingTimes[i + 1] - exitTimesLayup[i])
        idleTimesLayup.append(idleTimeLayup)
        idleTimeCompact = max(0, exitTimesLayup[i] - exitTimesCompact[i])
        idleTimesCompact.append(idleTimeCompact)

    numReject = numpy.logical_or(rejectPartsLayup, rejectPartsCompact)
    #print(numReject)
    resourceUtilLayup = (environment.now - sum(idleTimesLayup)) / environment.now
    resourceUtilCompact = (environment.now - sum(idleTimesLayup)) / environment.now
    # output relevant values
    simMeanWaitingTimes.append(statistics.mean(waitingTimesLayup))
    #simVarianceWaitingTimes.append(statistics.variance(waitingTimesLayup))
    simIdleTimes.append(sum(idleTimesLayup))
    simTotalQueueTime.append(sum(waitingTimesLayup))
    simMinQueue.append(min(queueLengthsLayup))
    simMaxQueue.append(max(queueLengthsLayup))
    simMeanQueue.append(statistics.mean(queueLengthsLayup))
    simResourceUtil.append(resourceUtilLayup)
    simResourceComp.append(resourceUtilCompact)
    simRejectRatio.append(sum(numReject) / entity_number)
    costRun = environment.now/60 * float(worker_hourly) * int(server1_capacity)\
              + environment.now/60 * float(worker_hourly) * int(server2_capacity)
    simCost.append(costRun)
    completedEntites = entity_number - entity_number * (sum(numReject) / entity_number)
    simCostPerEntity.append(costRun / completedEntites)
    simTotalTime.append(environment.now)

    interarrivalTimes.clear()
    serviceTimesLayup.clear()
    serviceTimesCompact.clear()
    rejectPartsLayup.clear()
    rejectPartsCompact.clear()
    waitingTimesLayup.clear()
    waitingTimesCompact.clear()
    queueLengthsLayup.clear()
    queueLengthsCompact.clear()
    idleTimesLayup.clear()
    idleTimesCompact.clear()
    exitTimesLayup.clear()
    exitTimesCompact.clear()
    arrivingTimes.clear()

costMean = statistics.mean(simCost)
totMeanWaitingTimes = statistics.mean(simMeanWaitingTimes)
print('Mean of the waiting times: %s' % totMeanWaitingTimes)

#totVarianceWaitingTimes = statistics.mean(simVarianceWaitingTimes)
#print('Variance of the waiting times: %s' % totVarianceWaitingTimes)

totIdleTimes = statistics.mean(simIdleTimes)
print('Mean of the idle times: %s' % totIdleTimes)

totQueueTime = statistics.mean(simTotalQueueTime)
print('Mean of the queue times: %s' % totQueueTime)

totRejectRatio = statistics.mean(simRejectRatio)
print('Mean of rejection ratios are %s' % totRejectRatio)

completedEntites = entity_number - entity_number * totRejectRatio
print('Completed Entites: %s' % str(completedEntites))

costPerEntity = costMean / completedEntites

print(f'Mean Cost per Entity with the {operator_name}: {costPerEntity}')
totMinQueue = statistics.mean(simMinQueue)
print('Mean of the minimum queue length: %s' % totMinQueue)
#
totMaxQueue = statistics.mean(simMaxQueue)
print('Mean of the maximum queue length: %s' % totMaxQueue)
#
totMeanQueue = statistics.mean(simMeanQueue)
print('Mean of the mean queue lengths: %s' % totMeanQueue)

totResourceUtil = statistics.mean(simResourceUtil)
print('Mean of the resource utilization: %s' % totResourceUtil)

print("cost is:")
print(simCost)

print(f'The mean total time for {operator_name} to produce {completedEntites} is {statistics.mean(simTotalTime)}')



#plot relavant figures
plt.subplot(2,1,1)
plt.hist(simCost, 25, density = True)
plt.ylabel('')
plt.xlabel('Total Cost')
plt.title('PDF of Total Cost')
plt.subplot(2,1,2)
plt.hist(simCost, 1000, density = True, cumulative = True)
plt.ylabel('')
plt.xlabel('Total Cost')
plt.title('CDF of Total Cost')
plt.tight_layout()
plt.savefig('total_cost.png')

plt.clf()
plt.subplot(2,1,1)
plt.hist(simCostPerEntity, 25, density = True)
plt.ylabel('Probability of Occurrence')
plt.xlabel('Cost per Entity')
plt.title('PDF of Cost per Entity: Cobot')
plt.subplot(2,1,2)
plt.hist(simCostPerEntity, 1000, density = True, cumulative = True)
plt.ylabel('Cumulative Probability of Occurrence')
plt.xlabel('Cost per Entity')
plt.title('CDF of Cost per Entity: Cobot')
plt.tight_layout()
plt.savefig('cost_per_entity.png')
