# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:59:36 2014

@author: Qian Wan
"""

import pytest
import test_1 
import test_2
import test_3

class TestFunctions:

    def test_task1(self):
        
        test_1.Find_popular() # find the most and least popular terms
        test_1.Find_term('guitar') # find types of guiar included

    def test_task2(self):
              
        test_2.Extract_topics(10,5) # extract top 10 topics, each contains 5 words
        test_2.Extract_Groups(10) # extract 10 groups by K means
        
    def test_task3(self):
        
        test_3.Predict() # print out classification results

pytest.main()
