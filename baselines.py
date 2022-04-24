import gzip
from collections import defaultdict
from sklearn import linear_model
import csv


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


def readCSV(path):
    f = gzip.open(path, 'rt')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header, l))
        yield d['user_id'], d['recipe_id'], d


def read_csv(path):
    with open(path) as f:
        c = csv.reader(f)
        header = next(c)
        for l in c:
            d = dict(zip(header, l))
            yield d['user_id'], d['recipe_id'], d


### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)

for user, recipe, d in readCSV("trainInteractions.csv.gz"):
    r = int(d['rating'])
    allRatings.append(r)
    userRatings[user].append(r)

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rated.txt", 'w')
for l in open("stub_Rated.txt"):
    if l.startswith("user_id"):
        #header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if u in userAverage:
        predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
    else:
        predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-cook baseline: just rank which recipes are popular and which are not, and return '1' if a recipe is among the top-ranked

recipeCount = defaultdict(int)
totalCooked = 0

for user, recipe, _ in readCSV("trainInteractions.csv.gz"):
    recipeCount[recipe] += 1
    totalCooked += 1

mostPopular = [(recipeCount[x], x) for x in recipeCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
# ic: item count, i: item
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalCooked / 2:
        break

predictions = open("predictions_Made.txt", 'w')
for l in open("stub_Made.txt"):
    if l.startswith("user_id"):
        #header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if i in return1:
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()

### Cook-time prediction baseline: Regress based on the length of the instructions

X = []
y = []

for d in readGz("trainRecipes.json.gz"):
    X.append([1, len(d['steps'])])
    y.append(d['minutes'])

mod = linear_model.LinearRegression()
mod.fit(X, y)

predictions = open("predictions_Minutes.txt", 'w')
predictions.write("recipe_id,prediction\n")
for d in readGz("testRecipes.json.gz"):
    x = [1, len(d['steps'])]
    pred = mod.predict([x])[0]
    predictions.write(d['recipe_id'] + ',' + str(pred) + '\n')

predictions.close()
