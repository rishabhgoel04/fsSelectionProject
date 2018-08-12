from django.urls import path
from django.contrib.auth.views import login
from . import views

app_name = 'featureselection'
urlpatterns = [
    path('', views.index, name='index'),
    path('add_task/', views.add_task, name='add_task'),
    path('task_result/<int:task_id>', views.task_result, name='task_result'),
    path('show_result/<int:task_id>', views.show_result, name='show_result'),
    path('show_user_result/<int:task_id>', views.show_user_result, name='show_user_result'),
    path('show_best_result/<int:task_id>', views.show_best_result, name='show_best_result'),
    path('export_result/<int:task_id>/<int:index1>/<int:index2>', views.export_result, name='export_result'),
    path('analyze_user_choice/<int:task_id>', views.analyze_user_choice, name='analyze_user_choice'),
    path('delete_own_result/<int:r_id>', views.delete_own_result, name='delete_own_result'),
    path('show_history/<username>', views.show_history, name='show_history'),
    path('delete_task/<int:task_id>/<username>', views.delete_task, name='delete_task'),
    path('download/<int:task_id>/<int:index1>/<int:index2>', views.download, name='download'),
    path('download_bestsubset/<int:task_id>', views.download_bestsubset, name='download_bestsubset'),
    path('choose_result/<int:task_id>/<int:index1>/<int:index2>', views.choose_result, name='choose_result'),
]