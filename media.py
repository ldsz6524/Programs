import webbrowser

class Movie():
    '''This class provides a way to store movie related information'''
      
    def __init__(self, movie_title, poster_image, trailer_youtube):
        '''This method is uesd to initialise the object'''
        
        '''Line 8 to 10 are the definition of instance variables'''
        self.title = movie_title            
        self.poster_image_url = poster_image
        self.trailer_youtube_url = trailer_youtube   


    def show_trailer(self):
        '''Show the trailer of the movie'''
        webbrowser.open(self.trailer_youtube_url)
        
        
