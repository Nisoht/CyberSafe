import { useState } from 'react';
import { Button } from './ui/button';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { User, LogOut } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from './ui/alert-dialog';

const Header = () => {
  const { user, signOut } = useAuth();
  const [open, setOpen] = useState(false);

  const username =
  user?.user_metadata?.username ||
  user?.user_metadata?.full_name || 
  user?.user_metadata?.name || 
  "User";

  
  const handleSignOut = async () => {
    await signOut();
    setOpen(false);
  };

  return (
    <header className="w-full py-4 px-6 border-b bg-white">
      <div className={`${user ? "flex-col lg:flex-row" : "flex-row"} max-w-7xl mx-auto flex justify-between items-center`}>
        <div className={` flex items-center space-x-2`}>
          <div className="h-12 w-12 rounded-full bg-cybersafe-600 flex items-center justify-center">
            <img src="/logo.png" alt="Logo" className="h-10 w-10" />
          </div>
          <Link to="/">
            <h1 className="text-xl font-bold text-cybersafe-800">CyberSafe</h1>
          </Link>
        </div>
        <div className="flex items-center space-x-4">
          {user ? (
            <>
              <div className="flex items-center gap-2">  
                <User size={16} className="text-cybersafe-600" />
                <div className="flex flex-col items-start">
                  <span className="text-sm font-medium text-gray-700">{username}</span>
                  <span className="text-xs text-gray-500">{user.email}</span>
                </div>
              </div>
              <AlertDialog open={open} onOpenChange={setOpen}>
                <AlertDialogTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="sm"
                    className="border-cybersafe-200 hover:bg-cybersafe-50 text-cybersafe-800 flex items-center gap-1"
                  >
                    <LogOut size={16} />
                    Sign Out
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Sign Out Confirmation</AlertDialogTitle>
                    <AlertDialogDescription>
                      Are you sure you want to sign out from your account?
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleSignOut}>Sign Out</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          ) : (
            <>
              <Button 
                variant="outline" 
                className="border-cybersafe-200 hover:bg-cybersafe-50 text-cybersafe-800"
                asChild
              >
                <Link to="/auth">Log In</Link>
              </Button>
              <Button className="bg-cybersafe-600 hover:bg-cybersafe-700 text-white" asChild>
                <Link to="/auth?tab=signup">Sign Up</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
