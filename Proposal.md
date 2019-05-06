

# Data Incubator Project Proposal

## Introduction

In sports, it is common for awards to be given out at the end of a season to recognize the best players. In the National Basketball Association (NBA), that award is an All-NBA designation. All-NBA awards are given to the 12 best players in the league, during that season, as selected by NBA analysts. In addition to
being an award, All-NBA is also commonly used while describing a player’s future potential as an athlete. For example, media members and pundits may refer to a rookie as having All-NBA potential or being an All-NBA player, without having had earned the award. Referring to rookies as potential all-NBA award winners occur when discussing the future value of a player or their current value as a trading piece for another player.

Understanding the future value of a rookie is vital to building a team and when developing trade packages. For example, if a team can be more confident of a rookie's future value compared to other rookies, they may decide to build around that player and exert resources in that player's development. Conversely, if a team is seeking to obtain a player, they can identify those individuals with high future value better than other teams and can put together beneficial trade packages.

Therefore it begs the question: is there a way we can predict a rookie's future value? There are several follow-up questions: how do you measure value? How far in the future do we want to know that players value? To keep this project within the scope of this challenge, we reference the NBA pundits language of deciding whether or not a player is of all-NBA caliber. Thus, the prediction task becomes one of whether or not the player will become an all-NBA within their career based on their rookie season stats.

## Prediction Task

The goal of this project, stated in the introduction, is to predict the career trajectory of a rookie player, or someone in their first season. The purpose is described above, but this section focuses on how to operationalize that prediction task. From the introduction we determine that the more specific goal is to predict whether or not a player will be an All-NBA award recipient within their career. To make this an task we can implement we need to put more definitions on it. Firstly, how long into a players career, after their first year, are we interested in predicting? Here we will arbitrarily select three time points: 10, 5, and 3 years after a players first season. 

The next question is who will our sample consist of? Will it be all first-seasons played across the entire history of the league? Or will be only use more recent data? For this problem we will utilize the entire history of the league when building our dataset. Therefore, the operationalization of the prediction task is the following: Given the stats of a players rookie season, can we predict whether or not they will win an all-NBA designation within 10, 5, or 3 years after their first season?