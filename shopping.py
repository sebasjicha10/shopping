import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    # Handle lables
    def handleLabels(boolean):
        if boolean == "TRUE":
            return 1
        else:
            return 0

    # Handle evidence
    def handleEvidence(evid):
        numeric_evidence = []

        # Column 1 Admi
        numeric_evidence.append(int(evid[0]))

        # Colum 2 Admi Duration
        numeric_evidence.append(float(evid[1]))

        # Colum 3 Info
        numeric_evidence.append(int(evid[2]))

        # Colum 4 Info Duration
        numeric_evidence.append(float(evid[3]))

        # Colum 5 Product
        numeric_evidence.append(int(evid[4]))

        # Colum 6 Product Duration
        numeric_evidence.append(float(evid[5]))

        # Colum 7 Bounce
        numeric_evidence.append(float(evid[6]))

        # Colum 8 Exit
        numeric_evidence.append(float(evid[7]))

        # Colum 9 Page Values
        numeric_evidence.append(float(evid[8]))

        # Colum 10 Special Day
        numeric_evidence.append(float(evid[9]))

        # Colum 11 Month
        months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        numeric_evidence.append(int(months.index(evid[10])))

        # Colum 12 Operating Systems
        numeric_evidence.append(int(evid[11]))

        # Colum 13 Browser
        numeric_evidence.append(int(evid[12]))

        # Colum 14 Region
        numeric_evidence.append(int(evid[14]))

        # Colum 15 Traffic Type
        numeric_evidence.append(int(evid[14]))

        # Colum 16 Visitor 
        if evid[15] == "Returning_Visitor":
            numeric_evidence.append(int(1))
        else:
            numeric_evidence.append(int(0))

        # Colum 17 Weekend
        if evid[16] == "TRUE":
            numeric_evidence.append(int(1))
        else:
            numeric_evidence.append(int(0))

        return numeric_evidence

    with open(filename) as filename:
        reader = iter(csv.reader(filename))
        next(reader)
        for row in reader:

            # Add labels
            labels.append(handleLabels(row[17]))

            # Add evidence
            evid = row[:-1]
            row_evidence = handleEvidence(evid)
            evidence.append(row_evidence)

    return (evidence, labels)
  

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    # Create the model
    kNeighbors_model = KNeighborsClassifier(n_neighbors=1)

    # Get the training fit
    result = kNeighbors_model.fit(evidence, labels)

    return result


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    sensitivity_count = 0
    specificity_count = 0

    total_positive = 0
    total_negative = 0

    for actual, predicted in zip(labels, predictions):
        
        # Count totals
        if actual == 1:
            total_positive += 1

        else:
            total_negative += 1

        # Count correct predictions
        if actual == predicted:
            if actual == 1:
                sensitivity_count += 1
            else:
                specificity_count += 1

    sensitivity = sensitivity_count / total_positive
    specificity = specificity_count / total_negative

    return sensitivity, specificity

    
if __name__ == "__main__":
    main()
