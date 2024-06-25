Note April 29, 2020
Trying to figure out where I was stuck June 26, 2019
I have some memory on being stuck on storing information 
sfgui.py and sfgui-working.py seem to be broken because uploading a file and plotting it does not work
I seem to have created upl.py just to test this.  Best to work there
Google searching the error indicates I'm probably trying to do something with a global variable, which is a no-no

Try this example of uploading and saving a file:
https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html
It worked! 
file_saver_example.py allows you to upload a file.  It is stored in app_uploaded_files/

Figured out the above, explained here:  https://community.plotly.com/t/unboundlocalerror-in-documentation-code-of-upload-component/33532
I wasn't uploading the right file type, so the first time it got to df was farther down, that threw an error. 
After uploading a 'csv'  I get no error.  Now I have to figure out how to plot that.

Now i'm working on upl.py, where I have two bits of code uploaded to try it.
May 3, 2020
  In upscatter.py, the file selector ingests the file, but I'm having trouble turning it into a pandas dataframe with columns of wav and flux.
May 4, 2020
  Got upscatter.py working -- can upload a .csv file and it makes a plot
  Now working in sfgui.py.  I get an error that 'contents' is a list (with only one element) and thus it can't be split because it isn't a string
  Not sure why because it is a string (I think) in upscatter.py.  Convert to string?  Check type?
May 5, 2020
  Figured it out. (it was because I was allowing multiple uploads -- this generates a list)
  Now sfgui.py works!
  Moving to sfgui.py sfgui-works.py
  Got store working to store output of SN types checklist.  Right now it is just printing what it is as a list.  But I need to read it out of storage.  Use the "on clicks" example.  
May 7
  sfgui.py:  Finally was able to get my supernova type selections to update in the store, then report them out
  Next to do: Create a callback with multiple inputs to the store.  Then create a list or possibly JSON that stores all these inputs for passing to sf.py.
Sept. 15 
  Got stuff pushed to GitHub -- see below
  Trying to decide if I need a true multipage app or can just do two tabs in a single app.  I created sfindex.py to try to be the main program tha has both the supernfit and supergraph tabs.  I modified sfgui.py to sftab.py.  But all I really did there is name @app to @sfapp (not even sure if that is necessary).  I was able to get most of the layout to come up in the tab, but the callbacks don't work, and some of the layouts that depend on callbacks don't work.  Next step -- maybe put everything from sfgui and sggui into a single sfindex file?

Feb 15
I got sfindex.py working so that it runs sftab and sgtab in tabs successfully with functioning callbacks.  I had to change some sgtab stuff (by commenting it out) because they changed DataTable.  Need to fix.  


GIT Stuff

moved stuff in superfit4 directory to "old"
git clone https://github.com/samanthagoldwasser25/GUIsuperfit
git remote -v [To check]
cd GUIsuperfit
cp ../old/sfgui.py .
cp ../old/sggui.py .
cp ../README.txt .
cp SN2019ein.csv .
git add sfgui.py sggui.py SN2019ein.csv 
git commit -m "First commit with sfgui.py and sggui.py"
git push origin master

TO UPGRADE DASH
pip install --upgrade dash
Now on 1.19.0 


August 4, 2021
Making a new superfit directory
cd /Users/ahowell/Dropbox/superfit4/
git clone https://github.com/samanthagoldwasser25/
cd superfit
git remote -v [To check]
cp ../GUIsuperfit/sfindex.py .
cp ../GUIsuperfit/sftab.py .
cp ../GUIsuperfit/sgtab.py .
cp ../GUIsuperfit/README.txt .
cp ../GUIsuperfit/SN2019ein.csv .
cp ../GUIsuperfit/SN2019ein.p02.sfo .
git add sftab.py sgtab.py sfindex.py SN2019ein.csv SN2019ein.p02.sfo
git commit -m "First commit with Andy's stuff - sfindex.py sftab.py sgtab.py SN2019ein.csv SN2019ein.p02.sfo"
git push origin master

I got: 
error: src refspec master does not match any.
error: failed to push some refs to 'https://github.com/samanthagoldwasser25/superfit'

Maybe because git branch says it is now named 
New-master

This is probably because master is still GUIsuperfit


August 5
Tried:
Installed template bank from Dropbox into bank directory.  This is not clear in the documentation.
Tried to run via:
python run.py
But this apparently requires numba.
pip install numba
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade astropy
pip install --upgrade PyAstronomy
pip install --upgrade pathlib

