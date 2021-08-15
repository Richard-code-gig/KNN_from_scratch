"""
Listing 1. Knn_classifier.py
Problem: Get the computer to perform a knn classifier. 
Target Users: Me
Target System: Windows 
Interface: Script
Functional Requirements: Draw a geolocation map of US showing state boundaries and print out a KNN clasifier message 
User must have a states.txt file and point to the file path. 
Testing: Simple run test - expecting a map to appear and when closed a message should appear.
Maintainer: https:www.datasudo.com
"""

#I designed everything from scratch so the only external module needed is matplotlib

import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt

"""This program classifies a given data using KNN classifier built from scratch."""

"""We need some functions that can manipulate matrices (vectors), so we will first define those."""

def vector_subtract(x, y):
    """subtracts corresponding elements in two matrices"""
    return [x_i - y_i for x_i, y_i in zip(x, y)]

def dot(x, y):
    """multiplies the components of two matrices and sum the values"""
    return sum(x_i * y_i for x_i, y_i in zip(x, y))

def sum_of_squares(x):
    """Squares the components of a matrix and sums the result"""
    return dot(x, x)

def magnitude(x):
    """Takes the square root of the function describe above"""
    return sqrt(sum_of_squares(x))

def distance(x, y):
    """Calculate the distance between two vectors"""  
    return magnitude(vector_subtract(x, y))

"""We need an intelligent function that can help vote the best of the rest from a close enough data point in case of a tie"""
def majority_vote(data):
    """assumes that data is ordered from nearest to farthest"""
    count_of_vote = Counter(data)
    only_winner, count_of_winner =count_of_vote.most_common(1)[0]
    num_winners = len([count for count in count_of_vote.values() if count == count_of_winner])
    if num_winners == 1:
        return only_winner # return the obvious unique winner
    else:
        return majority_vote(data[:-1]) #If no unique winner, perform a recursive function till we are left with one winner and then return it

def lat_lon_partition():
    """This partions the cordidnates of the state bondaries in the text file."""
    with open(r'c:/Users/richa/OneDrive/states.txt') as f: #states.txt file contains US state boundary coordinates. 
        lat = []
        lat_lon = []
        for lines in f.readlines():
            if lines.startswith("</state>"):
                for point_1, point_2 in zip(lat, lat[1:]):
                    lat_lon.append((point_1, point_2))
                lat = [] #After getting all boundary coordinates for a single state, instantiate back to an attempting list to get for another state.
            line = lines.strip()
            if line.startswith('<point'):
                x = line.split()[1].split('=')[1].strip('"')
                y = line.split()[2].split('=')[1].strip('"/>')
                lat.append((float(y), float(x)))
    return lat_lon        
        

def plot_US_state_boundaries(color='0.9'):
    """Plots a map using state border cordinates"""
    for (lon_1, lat_1), (lon_2, lat_2) in lat_lon_partition():
        plt.plot([lon_1, lon_2], [lat_1, lat_2], color=color)


"""This contain some hypothetical data of users in the US showing their coordinates and preferred languages. Ww will use this to plot our map and KNN classifier"""

US_cities = [(-86.75,33.5666666666667,'Python'),(-88.25,30.6833333333333,'Python'),
(-112.016666666667,33.4333333333333,'Java'),(-110.933333333333,32.1166666666667,'Java'),
(-92.2333333333333,34.7333333333333,'R'),(-121.95,37.7,'R'),(-118.15,33.8166666666667,'Python'),
(-118.233333333333,34.05,'Java'),(-122.316666666667,37.8166666666667,'R'),(-117.6,34.05,'Python'),
(-116.533333333333,33.8166666666667,'Python'),(-121.5,38.5166666666667,'R'),
(-117.166666666667,32.7333333333333,'R'),(-122.383333333333,37.6166666666667,'R'),
(-121.933333333333,37.3666666666667,'R'),(-122.016666666667,36.9833333333333,'Python'),
(-104.716666666667,38.8166666666667,'Python'),(-104.866666666667,39.75,'Python'),
(-72.65,41.7333333333333,'R'),(-75.6,39.6666666666667,'Python'),(-77.0333333333333,38.85,'Python'),
(-80.2666666666667,25.8,'Java'),(-81.3833333333333,28.55,'Java'),
(-82.5333333333333,27.9666666666667,'Java'),(-84.4333333333333,33.65,'Python'),
(-116.216666666667,43.5666666666667,'Python'),(-87.75,41.7833333333333,'Java'),
(-86.2833333333333,39.7333333333333,'Java'),(-93.65,41.5333333333333,'Java'),
(-97.4166666666667,37.65,'Java'),(-85.7333333333333,38.1833333333333,'Python'),
(-90.25,29.9833333333333,'Java'),(-70.3166666666667,43.65,'R'),(-76.6666666666667,39.1833333333333,'R'),
(-71.0333333333333,42.3666666666667,'R'),(-72.5333333333333,42.2,'R'),
(-83.0166666666667,42.4166666666667,'Python'),(-84.6,42.7833333333333,'Python'),
(-93.2166666666667,44.8833333333333,'Python'),(-90.0833333333333,32.3166666666667,'Java'),
(-94.5833333333333,39.1166666666667,'Java'),(-90.3833333333333,38.75,'Python'),
(-108.533333333333,45.8,'Python'),(-95.9,41.3,'Python'),(-115.166666666667,36.0833333333333,'Java'),
(-71.4333333333333,42.9333333333333,'R'),(-74.1666666666667,40.7,'R'),(-106.616666666667,35.05,'Python'),
(-78.7333333333333,42.9333333333333,'R'),(-73.9666666666667,40.7833333333333,'R'),
(-80.9333333333333,35.2166666666667,'Python'),(-78.7833333333333,35.8666666666667,'Python'),
(-100.75,46.7666666666667,'Java'),(-84.5166666666667,39.15,'Java'),(-81.85,41.4,'Java'),
(-82.8833333333333,40,'Java'),(-97.6,35.4,'Python'),(-122.666666666667,45.5333333333333,'Python'),
(-75.25,39.8833333333333,'Python'),(-80.2166666666667,40.5,'Python'),
(-71.4333333333333,41.7333333333333,'R'),(-81.1166666666667,33.95,'R'),
(-96.7333333333333,43.5666666666667,'Python'),(-90,35.05,'R'),(-86.6833333333333,36.1166666666667,'R'),
(-97.7,30.3,'Python'),(-96.85,32.85,'Java'),(-95.35,29.9666666666667,'Java'),
(-98.4666666666667,29.5333333333333,'Java'),(-111.966666666667,40.7666666666667,'Python'),
(-73.15,44.4666666666667,'R'),(-77.3333333333333,37.5,'Python'),(-122.3,47.5333333333333,'Python'),
(-89.3333333333333,43.1333333333333,'R'),(-104.816666666667,41.15,'Java')]

# we want to give each language a unique marker and color
markers = { "Java" : "o", "Python" : "s", "R" : "^" }
colors = { "Java" : "r", "Python" : "b", "R" : "g" }
plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

for longitude, latitude, language in US_cities:
    plots[language][0].append(longitude)
    plots[language][1].append(latitude)

for language, (x,y) in plots.items():
    plt.scatter(x, y, color=colors[language], marker=markers[language], label=language, zorder=10)

plt.legend(loc=0)
plt.axis([-135,-65,20,60])
plot_US_state_boundaries(color='0.7') #call this function to plot the map of the state borders.
plt.title("Favorite Programming Languages")
plt.show() #The show method will output the graph which we must close to view the result of our KNN classsifier

def knn_classifier(k, raw_points, new_point):
    sort_by_dist = sorted(raw_points, key=lambda point: distance(point, new_point)) #Used the distance function described above to return a new sequence of sorted data
    k_nearest_labels = [label for _, _, label in sort_by_dist[:k]] #find the data closest to k
    return majority_vote(k_nearest_labels) # and let them vote

for k in [1, 3, 5, 7]: #For some reasons k is usually odd numbers.
    num_correct = 0
    for city in US_cities:
        long, lat, actual_language = city
        other_US_cities = [other_city for other_city in US_cities if other_city != city]
        predicted_language = knn_classifier(k, other_US_cities, (long, lat)) #sort using difference between longitude and latitude
        if predicted_language == actual_language:
            num_correct += 1
            accuracy = (num_correct/len(US_cities)) * 100 
            rounded_accuracy =  round(accuracy, 2) 
            if k == 1:
                a = num_correct
            elif k == 3:
                b = num_correct
            elif k == 5:
                c = num_correct
            else:
                d = num_correct

    print(f'{k} neighbor: {num_correct} correct out of {len(US_cities)}. I.e {str(rounded_accuracy)}% accuracy')
print()
max_val = max(a,b,c,d)
if a == max_val:
    print('1 nearest neighbor is the best classifier for this data')
elif b == max_val:
    print('3 nearest neighbor is the best classifier for this data')
elif c == max_val:
    print('5 nearest neighbor is the best classifier for this data')
else:
    print('7 nearest neighbor is the best classifier for this data')