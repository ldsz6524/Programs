import fresh_tomatoes

import media



'''The following codes are uesd to generate objects(movies)'''
god_father = media.Movie('The Godfather',
                         'https://i.pinimg.com/736x/d3/7b/66/d37b66b2d670ded0887f1414d528e9f7--godfather-movie-watch-the-godfather.jpg',
                         'https://www.youtube.com/watch?v=sY1S34973zA')


forrest_gump = media.Movie('Forrest Gump',
                           'http://myfilmhd.me/uploads/mini/250x370/75/1492948770-22224803-forrest-gamp.jpg',
                           'https://www.youtube.com/watch?v=u7x4QwzLRaI')



first_blood = media.Movie('First Blood',
                          'https://i.pinimg.com/originals/76/b3/04/76b3046212cf9b44d3aaebe125c5e72b.jpg',
                          'https://www.youtube.com/watch?v=IAqLKlxY3Eo')



apple_of_my_eye = media.Movie('You Are The Apple Of My Eye',
                              'https://gss0.baidu.com/-fo3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/0e2442a7d933c8958fc321dfd11373f0830200d0.jpg',
                              'https://www.youtube.com/watch?v=v5H6wE47FrI')



a_better_tomorrow = media.Movie('A Better Tomorrow',
                                'http://is1.mzstatic.com/image/thumb/Video5/v4/ca/3c/e9/ca3ce95f-1e21-f07a-2f32-6b2a5f90202e/source/1200x630bb.jpg',
                                'https://www.youtube.com/watch?v=M7ZBsUf986E')



roman_holiday = media.Movie('Roman Holiday',
                            'https://images-na.ssl-images-amazon.com/images/I/518EPG6E91L._SY445_.jpg',
                            'https://www.youtube.com/watch?v=eIFo0txAvuE')



movies = [god_father, forrest_gump, first_blood,
          apple_of_my_eye, a_better_tomorrow, roman_holiday]

'''Generate .html file or website page'''
fresh_tomatoes.open_movies_page(movies)





































