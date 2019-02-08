This simple POC console program will enumerate through text files in a given folder to capture the folder/file names and break down the file text into chunks which can be passed to the Microsoft Text Analytics API. 
The Text Analytics API can accept text up to 5000 characters per input. Each text input will count toward your call limit in the free tier (5000 calls per month). 

The resulting key phrase analyses of the chunks are consolidated to be saved with the original file information as JSON, to a path corresponding to the original text file in an output directory of your choosing. 

TODO: Add your own Subscription key for your own Cognitive Services Resource in your Azure Portal

This project is using early versions of the Microsoft nuget packages, so you may wish to update the packages for whatever new functionality has been released and fix the code as needed.