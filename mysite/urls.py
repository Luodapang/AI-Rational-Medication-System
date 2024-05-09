"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from extract import views as ins
from predict import views as rm
from controllers import views
from django.views import static ##新增
from django.conf import settings ##新增
from django.conf.urls import url ##新增
from controllers import KnowledgeSearch, DrugRisk, DrugEco

urlpatterns = [
    path('admin/', admin.site.urls),
    path('DrugEco', rm.DrugEco),
    path('', views.homepage),
    path('index', views.index),
    # path('ileft/', ins.ileft),
    # path('iright/', ins.iright),
    # path('ihomepage', ins.ihomepage),
    # path('ihomepage/ileft', ins.ileft),
    # path('ihomepage/iright', ins.iright),
    path('drugRisk', views.drugRisk),
    path('diagSearch', views.diagSearch),
    path('medSearch', views.medSearch),
    path('patternSearch', views.patternSearch),
    path('prescriptionSearch', views.prescriptionSearch),
    url('DrugEco/evaluateEco', DrugEco.evaluateEco, name='DrugEco/evaluateEco'),
    url('KnowledgeSearch/diagSearch', KnowledgeSearch.diagSearch, name='KnowledgeSearch/diagSearch'),
    url('KnowledgeSearch/medSearch', KnowledgeSearch.medSearch, name='KnowledgeSearch/medSearch'),
    # url('KnowledgeSearch/patternSearch', KnowledgeSearch.patternSearch, name='KnowledgeSearch/patternSearch'),
    # url('KnowledgeSearch/prescriptionSearch',KnowledgeSearch.prescriptionSearch,name='KnowledgeSearch/prescriptionSearch'),
    url('DrugRisk/evaluateRisk', DrugRisk.evaluateRisk, name='DrugRisk/evaluateRisk'),
    url('DrugRisk/getInfo', DrugRisk.getInfo, name='DrugRisk/getInfo'),
    url('DrugRisk/matchDiag', DrugRisk.matchDiag, name='DrugRisk/matchDiag'),
    url(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'),
]
