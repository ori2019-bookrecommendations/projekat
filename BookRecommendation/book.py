class Book:

    def __init__(self, id, name, authors, average_rating, picture):
        self.id = id
        self.name = name
        self.authors = authors,
        self.average_rating = average_rating
        self.picture = picture

    def __str__(self):
        return str(self.name + ", %s" % self.authors)

    def getId(self):
        return self.id

    def setId(self, id):
        self.id = id

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getAuthors(self):
        return self.authors

    def setAuthors(self, authors):
        self.authors = authors

    def getAverageRating(self):
        return self.average_rating

    def setAverageRating(self, average_rating):
        self.average_rating = average_rating

    def getPicture(self):
        return self.picture

    def setPicture(self, picture):
        self.picture = picture
