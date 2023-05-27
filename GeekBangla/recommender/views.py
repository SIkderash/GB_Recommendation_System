from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

def write_2d_array_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def load_csv_as_array_without_row_index_and_column_headers(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    data = np.delete(data, 0, axis=1)
    return data

def load_csv_as_array(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

# Create your views here.
@api_view(['POST'])
@csrf_exempt
def train(request):
    if request.method=='POST':  
        data = JSONParser().parse(request)
        file_path = data["file_path"]
        dataset = load_csv_as_array_without_row_index_and_column_headers(file_path)
        print("dataset\n" , dataset)

        # Calculate the user similarity matrix
        user_similarity = cosine_similarity(dataset)
        write_2d_array_to_csv(user_similarity, "user_similarity.csv")
        return JsonResponse("User Similarity Calculated", safe = False)


@api_view(['POST'])
@csrf_exempt
def recommend(request):
    if request.method=='POST':                                  
        users_data=JSONParser().parse(request)

        target_user = users_data['id']

        dataset = load_csv_as_array_without_row_index_and_column_headers('dataset.csv')
        user_similarity = load_csv_as_array('user_similarity.csv')

        # Find the most similar users to the target user
        similar_users = np.argsort(user_similarity[target_user])[::-1][1:]
        similar_users = similar_users[:10]

        # Recommend categories based on similar users' preferences
        recommendations = []
        for category in range(dataset.shape[1]-1):
            if dataset[target_user, category] == 0:  # User has not rated this category
                category_rating = 0
                similarity_sum = 0
                for user in similar_users:
                    if dataset[user, category] != 0:  # Similar user has rated this category
                        category_rating += dataset[user, category] * user_similarity[target_user, user]
                        similarity_sum += user_similarity[target_user, user]
                if similarity_sum > 0:
                    category_rating /= similarity_sum
                recommendations.append((category, category_rating))

        # Sort the recommendations by rating (descending order)
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        # Print the recommendations
        print("Recommendations for User", target_user + 1)
        for recommendation in recommendations:
            category, category_rating = recommendation
            print("Category", category + 1, "- Rating:", category_rating)
        return JsonResponse(recommendations, safe = False)


@api_view(['GET'])
@csrf_exempt
def test(request):
    if request.method=='GET':
        return JsonResponse("Test Success", safe = False)        

